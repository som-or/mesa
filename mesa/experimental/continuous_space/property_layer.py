from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import scipy.ndimage as ndimage

# Types
ContinuousPos = Sequence[float]
Index = tuple[int, ...]
MaskFn = Callable[[np.ndarray], np.ndarray]


def _ufunc_requires_additional_input(ufunc: np.ufunc) -> bool:
    # numpy ufuncs expose .nin = number of inputs
    return ufunc.nin > 1


class PropertyLayer:
    """A raster property layer over a continuous domain.

    Args:
        name: Name of the layer.
        bounds: Array-like with shape (ndims, 2): [[min0, max0], [min1, max1], ...]
        resolution: Cells per unit distance (must be > 0).
        default_value: Fill value.
        dtype: Element dtype.
    """

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value) -> None:
        # same pattern as discrete Mesa PropertyLayer
        self._data[:] = value

    def __init__(
        self,
        name: str,
        bounds: Any,
        resolution: float = 1.0,
        default_value: Any = 0.0,
        dtype: type | np.dtype = float,
    ) -> None:
        if resolution <= 0:
            raise ValueError("resolution(cells per unit) must be > 0.")

        self.name = name
        self.bounds = np.asanyarray(bounds, dtype=float)
        # if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
        #     raise ValueError("bounds must have shape (ndims, 2).")

        self.ndims = int(self.bounds.shape[0])
        self.resolution = float(resolution)

        mins = self.bounds[:, 0]
        maxs = self.bounds[:, 1]

        # if not np.all(maxs > mins):
        #     raise ValueError("Each bounds row must satisfy max > min.")

        self.size = maxs - mins  # size of continuous space 

        # Raster shape: ceil(size * cells_per_unit)
        self.shape: tuple[int, ...] = tuple(int(math.ceil(s * self.resolution)) for s in self.size)

        # same as discrete version)
        try:
            if dtype(default_value) != default_value:
                warnings.warn(
                    f"Default value {default_value} will lose precision when converted to "
                    f"{getattr(dtype, '__name__', str(dtype))}.",
                    UserWarning,
                    stacklevel=2,
                )
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Default value {default_value} is incompatible with dtype={getattr(dtype, '__name__', str(dtype))}."
            ) from e

        self._data = np.full(self.shape, default_value, dtype=dtype)

   
    @classmethod
    def from_data(
        cls,
        name: str,
        data: np.ndarray,
        *,
        bounds: Any,
        resolution: float | None = None,
    ) -> "PropertyLayer":
        
        bounds_arr = np.asanyarray(bounds, dtype=float)
        # if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
        #     raise ValueError("bounds must have shape (ndims, 2).")

        size = bounds_arr[:, 1] - bounds_arr[:, 0]
        if data.ndim != bounds_arr.shape[0]:
            raise ValueError("data.ndim must match number of dimensions in bounds.")

        if resolution is None:
            inferred = np.array(data.shape, dtype=float) / size
            if not np.allclose(inferred[1], inferred[0]):
                raise ValueError(
                    f"Non-uniform inferred resolution across axes: {inferred}. "
                    "Provide an explicit resolution to avoid ambiguity."
                )
            resolution = float(inferred[0])
            
        layer = cls(
            name=name,
            bounds=bounds_arr,
            resolution=float(resolution),
            default_value=0,
            dtype=data.dtype.type,
        )
        if layer.shape != tuple(data.shape):
            raise ValueError(
                f"data.shape {data.shape} does not match expected shape {layer.shape} "
            )
        
        layer.data = data.copy()
        return layer

    
    def pos_to_index(self, pos: ContinuousPos,) -> Index:

        p = np.asanyarray(pos, dtype=float)
        if p.shape != (self.ndims,):
            raise ValueError(f"pos must have length {self.ndims}.")

        mins = self.bounds[:, 0]
        raw = np.floor((p - mins) * self.resolution).astype(int)

        if np.any(raw < 0) or np.any(raw >= np.array(self.shape)):
            raise IndexError("Position maps outside layer grid")
        
        for i, dim in enumerate(self.shape):
            if raw[i] < 0:
                raw[i] = 0
            elif raw[i] >= dim:
                raw[i] = dim - 1

        return tuple(int(x) for x in raw)
        
    def index_to_pos(self, index: Sequence[int], *, center: bool = True) -> tuple[float, ...]:
        """
        If center=True, returns the center of the raster cell; otherwise returns cells upper left corner.
        """
        idx = np.asanyarray(index, dtype=float)
        if idx.shape != (self.ndims,):
            raise ValueError(f"index must have length {self.ndims}.")

        mins = self.bounds[:, 0]
        offset = 0.5 if center else 0.0
        pos = mins + (idx + offset) / self.resolution
        return tuple(float(x) for x in pos)


    # def set_cells(self, value: Any, condition: MaskFn | None = None) -> None:
    #     """Set cells to a value (optionally where a mask condition holds)."""
    #     if condition is None:
    #         self.data[:] = value
    #         return

    #     mask = condition(self.data)
    #     if mask.shape != self.data.shape or mask.dtype != bool:
    #         raise ValueError("condition must return a boolean mask with the same shape as data.")
    #     self.data[mask] = value

    def modify_cells(
        self,
        operation: Callable | np.ufunc,
        value: Any = None,
        condition: MaskFn | None = None,
    ) -> None:
        """Modify cells using an operation (ufunc or callable), optionally masked."""
        if condition is None:
            mask = slice(None)
            target = self.data
        else:
            mask_arr = condition(self.data)
            if mask_arr.shape != self.data.shape or mask_arr.dtype != bool:
                raise ValueError("condition must return a boolean mask with the same shape as data.")
            mask = mask_arr
            target = self.data[mask]

        if isinstance(operation, np.ufunc):
            if _ufunc_requires_additional_input(operation):
                if value is None:
                    raise ValueError("This ufunc requires an additional input value.")
                self.data[mask] = operation(target, value)
            else:
                self.data[mask] = operation(target)
        else:
            # Expect operation to accept ndarray and return ndarray (vectorized). No np.vectorize in core path.
            out = operation(target) if value is None else operation(target, value)
            self.data[mask] = out

    def select_cells(self, condition: MaskFn, return_list: bool = True):
        """Select cells by a value-based mask condition."""
        mask = condition(self.data)
        if mask.shape != self.data.shape or mask.dtype != bool:
            raise ValueError("condition must return a boolean mask with the same shape as data.")
        if return_list:
            return list(zip(*np.where(mask)))
        return mask

    def aggregate(self, operation: Callable[[np.ndarray], Any]) -> Any:
        """Aggregate over the layer data (e.g., np.sum, np.mean)."""
        return operation(self.data)

    def deposit(
        self,
        pos: Sequence["ContinuousPos"] | "ContinuousPos",
        value: Callable[["ContinuousPos"], float] | Sequence[float] | int | float,
        mode: str = "set",
        alpha: float | None = None,
    ) -> None:
        """
        value
            - scalar (int/float): same value for every position
            - callable: value(pos) computed per position
            - sequence: one value per position
        mode
            How to combine with existing cell values:
            - "set": overwrite cell value with new value
            - "add": add new value to existing cell value
            - "blend": weighted average: new = (1-alpha)*old + alpha*incoming
        alpha
            Used only for mode="blend". Must be in [0, 1].
        """
        if isinstance(pos, ContinuousPos):
            positions = [pos]
        else:
            positions = list(pos)

        if not positions:
            return
        
        indices = [self.pos_to_index(p) for p in positions]


        if callable(value):
            values = np.asarray([float(value(p)) for p in positions], dtype=float)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            values = np.full(len(positions), float(value), dtype=float)
        else:
            values = np.asarray(list(value), dtype=float)
            if len(values) != len(positions):
                raise ValueError("value length doesn't match pos length")

        # for one position
        if len(indices) == 1:
            idx = indices[0]
            v = float(values[0])
            if mode == "set":
                self.data[idx] = v
            elif mode == "add":
                self.data[idx] += v
            elif mode == "blend":
                if alpha is None:
                    raise ValueError("alpha must be provided for mode='blend'")
                if not (0.0 <= alpha <= 1.0):
                    raise ValueError("alpha must be in [0, 1]")
                self.data[idx] = (1.0 - alpha) * self.data[idx] + alpha * v
            else:
                raise ValueError(f"Unknown mode: {mode!r}")
            return

        # list of positions 
        indices_arr = np.asarray(indices, dtype=int)  # shape: (n, ndims)
        idx_tuple = tuple(indices_arr[:, d] for d in range(indices_arr.shape[1])) #(xs, ys)

        if mode == "set":
            self.data[idx_tuple] = values
            return

        if mode == "add":
            np.add.at(self.data, idx_tuple, values)
            return

        if mode == "blend":
            if alpha is None:
                raise ValueError("alpha must be provided for mode='blend'")
            if not (0.0 <= alpha <= 1.0):
                raise ValueError("alpha must be in [0, 1]")

            for idx, v in zip(indices, values):
                self.data[idx] = (1.0 - alpha) * self.data[idx] + alpha * float(v)
            return

        raise ValueError(f"Unknown mode: {mode!r}")
    
    def deposit_splat(
    self,
    pos: Sequence["ContinuousPos"] | "ContinuousPos",
    value: Callable[["ContinuousPos"], float] | Sequence[float] | int | float,
    mode: str = "set",
    alpha: float | None = None,
    kernel: str = "gaussian",
    spread: int = 1,
    torus: bool = False,
) -> None:
        """
        Deposit values onto the grid with local spatial spreading (splat).

        kernel
            One of {"gaussian", "linear", "epanechnikov"}.
        spread
            Radius in grid cells (integer).
        """

        if spread < 0:
            raise ValueError("spread must be >= 0")

        if isinstance(pos, ContinuousPos):
            positions = [pos]
        else:
            positions = list(pos)

        if not positions:
            return

        # Build values
        if callable(value):
            values = np.asarray([float(value(p)) for p in positions], dtype=float)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            values = np.full(len(positions), float(value), dtype=float)
        else:
            values = np.asarray(list(value), dtype=float)
            if len(values) != len(positions):
                raise ValueError("value length does not match pos length")

        if spread == 0:
            self.deposit(pos, value, mode, alpha)

        # offset window [-spread .. spread]
        side = 2 * spread + 1
        offsets = np.indices((side,) * self.ndims, dtype=int)
        for d in range(self.ndims):
            offsets[d] -= spread

        deltas = np.stack(offsets, axis=-1) / self.resolution
        dist2 = np.sum(deltas * deltas, axis=-1)

        # Compute kernel weights
        if kernel == "gaussian":
            sigma = spread / self.resolution if spread > 0 else 1.0
            weights = np.exp(-dist2 / (2 * sigma * sigma))
        elif kernel == "linear":
            d = np.sqrt(dist2)
            R = spread / self.resolution
            weights = np.clip(1 - d / R, 0.0, 1.0)
        elif kernel == "epanechnikov":
            R = spread / self.resolution
            weights = np.clip(1 - dist2 / (R * R), 0.0, 1.0)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        total = weights.sum()
        if total > 0:
            weights = weights / total

        # Apply splat for each position
        for p, v in zip(positions, values):
            center_idx = self.pos_to_index(p)

            # Map offsets to global indices
            target_idxs = []
            for d in range(self.ndims):
                idx = center_idx[d] + offsets[d]
                if torus:
                    idx %= self.shape[d]
                target_idxs.append(idx)

            if torus:
                contrib = v * weights
                if mode == "set":
                    self.data[tuple(target_idxs)] = contrib
                elif mode == "add":
                    self.data[tuple(target_idxs)] += contrib
                elif mode == "blend":
                    if alpha is None:
                        raise ValueError("alpha required for blend mode")
                    current = self.data[tuple(target_idxs)]
                    self.data[tuple(target_idxs)] = (
                        (1 - alpha) * current + alpha * contrib
                    )
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            else:
                # Mask out-of-bounds
                in_bounds = np.ones_like(dist2, dtype=bool)
                for d in range(self.ndims):
                    in_bounds &= (
                        (target_idxs[d] >= 0) & (target_idxs[d] < self.shape[d])
                    )

                clipped_idxs = [
                    target_idxs[d][in_bounds].astype(int)
                    for d in range(self.ndims)
                ]
                contrib = (v * weights)[in_bounds]

                if mode == "set":
                    self.data[tuple(clipped_idxs)] = contrib
                elif mode == "add":
                    np.add.at(self.data, tuple(clipped_idxs), contrib)
                elif mode == "blend":
                    if alpha is None:
                        raise ValueError("alpha required for blend mode")
                    current = self.data[tuple(clipped_idxs)]
                    self.data[tuple(clipped_idxs)] = (
                        (1 - alpha) * current + alpha * contrib
                    )
                else:
                    raise ValueError(f"Unknown mode: {mode}")

    def kernel_matrix(self, kernel_type: str, radius: float):
        side = 2 * radius + 1
        offsets = np.indices((side,) * self.ndims, dtype=int)
        for d in range(self.ndims):
            offsets[d] -= radius

        deltas = np.stack(offsets, axis=-1) / self.resolution
        dist2 = np.sum(deltas * deltas, axis=-1)

        if kernel_type == "gaussian":
            sigma = radius / self.resolution if radius> 0 else 1.0
            weights = np.exp(-dist2 / (2 * sigma * sigma))
        elif kernel_type == "linear":
            d = np.sqrt(dist2)
            R = radius  / self.resolution
            weights = np.clip(1 - d / R, 0.0, 1.0)
        elif kernel_type == "epanechnikov":
            R = radius / self.resolution
            weights = np.clip(1 - dist2 / (R * R), 0.0, 1.0)
        else:
            raise ValueError(f"Unknown kernel: {kernel_type}")

        total = weights.sum()
        if total > 0:
            weights = weights / total
        
        
    

    def get_neighborhood_mask(
        self,
        center_pos: ContinuousPos,
        radius: float,
        torus: bool = False,
    ) -> np.ndarray:
        
        if radius < 0:
            raise ValueError("radius must be >= 0")

        
        r_cells = int(math.ceil(radius * self.resolution))
        center_idx = self.pos_to_index(center_pos)

        mask = np.zeros(self.shape, dtype=bool)

        side = 2 * r_cells + 1
        offsets = np.indices((side,) * self.ndims, dtype=int)
        # [0..side-1] -> shift to [-r..r]
        for d in range(self.ndims):
            offsets[d] -= r_cells

        
        radius2 = float(radius) ** 2

        # grid-offsets to continuous-offsets
        deltas_cont = np.stack(offsets, axis=-1) / self.resolution  # shape (side, side, ndims)
        dist2 = np.sum(deltas_cont**2, axis=-1)  # x**2+ y**2
        local_inside = dist2 <= radius2  

        # local mask onto global mask
        target_indices = []
        for d, off in enumerate(offsets):
            idx = center_idx[d] + off
            if torus:
                idx = idx % self.shape[d]
            target_indices.append(idx.astype(int))


        mask[tuple(target_indices)] = local_inside
        return mask

    # --- Optional diffusion (global) ---
    def diffuse(
        self,
        distance: float,
        diffusion_rate: float = 1.0,
        *,
        torus: bool = False,
    ) -> None:
        """Diffuse the layer using a Gaussian filter.

        Args:
            distance: diffusion radius in continuous units.
            diffusion_rate: blending factor in [0,1]. 1.0 = fully replaced by diffused.
            torus: if True, wrap boundaries; else reflect.
        """
        if distance < 0:
            raise ValueError("distance must be >= 0")
        if not (0.0 <= diffusion_rate <= 1.0):
            raise ValueError("diffusion_rate must be in [0, 1].")

        # Convert continuous distance to sigma in cells.
        discrete_radius = distance * self.resolution
        sigma = discrete_radius / 3.0  # ~99% within 3 sigma
        if sigma <= 0:
            return

        mode = "wrap" if torus else "reflect"
        diffused = ndimage.gaussian_filter(self._data, sigma=sigma, mode=mode)

        self._data[:] = (self._data * (1.0 - diffusion_rate)) + (diffused * diffusion_rate)


class HasPropertyLayers:
    """Mixin for ContinuousSpace-like classes to manage raster PropertyLayers.

    Assumptions:
    - The host class has `dimensions` in Mesa ContinuousSpace format: shape (ndims, 2).
    - The host class may have `.torus` (bool).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._property_layers: dict[str, PropertyLayer] = {}

    def create_property_layer(
        self,
        name: str,
        resolution: float = 1.0,
        default_value: Any = 0.0,
        dtype: type | np.dtype = float,
    ) -> PropertyLayer:
        layer = PropertyLayer(
            name=name,
            bounds=self.dimensions,
            resolution=resolution,
            default_value=default_value,
            dtype=dtype,
        )
        self.add_property_layer(layer)
        return layer

    def add_property_layer(self, layer: PropertyLayer) -> None:
        if layer.name in self._property_layers:
            raise ValueError(f"Property layer {layer.name} already exists.")

        # Require same bounds as the space (PR-friendly default).
        space_bounds = np.asanyarray(self.dimensions, dtype=float)
        if space_bounds.shape != layer.bounds.shape or not np.allclose(space_bounds, layer.bounds):
            raise ValueError("Layer bounds must match the space bounds.")

        self._property_layers[layer.name] = layer

    def remove_property_layer(self, name: str) -> None:
        del self._property_layers[name]

    def get_property_layer(self, name: str) -> PropertyLayer:
        return self._property_layers[name]

    # Convenience get/set at continuous positions
    def get_value(self, property_name: str, pos: ContinuousPos) -> Any:
        layer = self._property_layers[property_name]
        idx = layer.pos_to_index(pos, clamp=True)
        return layer.data[idx]

    def set_value(self, property_name: str, pos: ContinuousPos, value: Any) -> None:
        layer = self._property_layers[property_name]
        idx = layer.pos_to_index(pos, clamp=True)
        layer.data[idx] = value

    # Parity forwarding for bulk ops
    def set_property(self, property_name: str, value: Any, condition: MaskFn | None = None) -> None:
        self._property_layers[property_name].set_cells(value, condition)

    def modify_property(
        self,
        property_name: str,
        operation: Callable | np.ufunc,
        value: Any = None,
        condition: MaskFn | None = None,
    ) -> None:
        self._property_layers[property_name].modify_cells(operation, value, condition)

    # Optional sugar: neighborhood mask via a specific property layer (resolution matters!)
    def get_neighborhood_mask(
        self,
        property_name: str,
        center_pos: ContinuousPos,
        radius: float,
        *,
        torus: bool | None = None,
    ) -> np.ndarray:
        layer = self._property_layers[property_name]
        use_torus = bool(self.torus) if torus is None else bool(torus)
        return layer.get_neighborhood_mask(center_pos, radius, torus=use_torus)

    # Optional attribute-style access (same UX as discrete, without collisions guarding)
    def __getattr__(self, name: str) -> Any:
        try:
            return self._property_layers[name]
        except KeyError as e:
            raise AttributeError(
                f"'{type(self).__name__}' object has no property layer called '{name}'"
            ) from e