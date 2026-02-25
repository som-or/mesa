from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from itertools import chain

# Types
ContinuousPos = Sequence[float]
Index = tuple[int, ...]
# MaskFn = Callable[[np.ndarray], np.ndarray]

def apply_mode_at(
    self,
    idx_tuple: tuple[np.ndarray, ...],
    incoming: np.ndarray,
    *,
    mode: str,
    alpha: float | None,
) -> None:
    if mode == "set":
        self.data[idx_tuple] = incoming
        return

    if mode == "add":
        np.add.at(self.data, idx_tuple, incoming)
        return

    if mode == "blend":
        if alpha is None:
            raise ValueError("alpha required for blend mode")

        # Sequential semantics (important if indices repeat)
        flat_coords = np.stack([a.ravel() for a in idx_tuple], axis=1)  # (n, ndims)
        flat_in = incoming.ravel()

        for coord, v in zip(flat_coords, flat_in):
            coord_t = tuple(int(c) for c in coord)
            self.data[coord_t] = (1.0 - alpha) * self.data[coord_t] + alpha * float(v)
        return

    raise ValueError(f"Unknown mode: {mode!r}")

def make_offsets(self, radius: float) -> np.ndarray:
    radius_cells = int(math.ceil(radius * self.resolution)) 
    if radius_cells < 0:
        raise ValueError("radius_cells must be >= 0")
    side = 2 * radius_cells + 1
    offsets = np.indices((side,) * self.ndims, dtype=int)
    return offsets - radius_cells 

def global_indices_from_center(
    self,
    center_pos: ContinuousPos,
    offsets: np.ndarray,
    torus: bool,
) -> np.ndarray:
    
    center_idx = self.pos_to_index(center_pos)
    target_indices = [] 
    for d, off in enumerate(offsets): 
        idx = center_idx[d] + off 
        if torus: 
            idx = idx % self.shape[d] 
            target_indices.append(idx.astype(int)) 
    
    return tuple(target_indices)


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
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (ndims, 2).")

        self.ndims = int(self.bounds.shape[0])
        self.resolution = float(resolution)

        mins = self.bounds[:, 0]
        maxs = self.bounds[:, 1]

        if not np.all(maxs > mins):
            raise ValueError("Each bounds row must satisfy max > min.")

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
        if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
            raise ValueError("bounds must have shape (ndims, 2).")

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

    # def modify_cells(
    #     self,
    #     operation: Callable | np.ufunc,
    #     value: Any = None,
    #     condition: MaskFn | None = None,
    # ) -> None:
    #     """Modify cells using an operation (ufunc or callable), optionally masked."""
    #     if condition is None:
    #         mask = slice(None)
    #         target = self.data
    #     else:
    #         mask_arr = condition(self.data)
    #         if mask_arr.shape != self.data.shape or mask_arr.dtype != bool:
    #             raise ValueError("condition must return a boolean mask with the same shape as data.")
    #         mask = mask_arr
    #         target = self.data[mask]

    #     if isinstance(operation, np.ufunc):
    #         if _ufunc_requires_additional_input(operation):
    #             if value is None:
    #                 raise ValueError("This ufunc requires an additional input value.")
    #             self.data[mask] = operation(target, value)
    #         else:
    #             self.data[mask] = operation(target)
    #     else:
    #         # Expect operation to accept ndarray and return ndarray (vectorized). No np.vectorize in core path.
    #         out = operation(target) if value is None else operation(target, value)
    #         self.data[mask] = out

    # def select_cells(self, condition: MaskFn, return_list: bool = True):
    #     """Select cells by a value-based mask condition."""
    #     mask = condition(self.data)
    #     if mask.shape != self.data.shape or mask.dtype != bool:
    #         raise ValueError("condition must return a boolean mask with the same shape as data.")
    #     if return_list:
    #         return list(zip(*np.where(mask)))
    #     return mask

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

        if callable(value):
            values = np.asarray([float(value(p)) for p in positions], dtype=float)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            values = np.full(len(positions), float(value), dtype=float)
        else:
            values = np.asarray(list(value), dtype=float)
            if len(values) != len(positions):
                raise ValueError("value length doesn't match pos length")

        
        indices = np.asarray([self.pos_to_index(p) for p in positions], dtype=int)
        idx_tuple = tuple(indices[:, d] for d in range(indices.shape[1])) #(xs, ys)

        
        apply_mode_at(idx_tuple, values, mode=mode, alpha=alpha)

        return 

    
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

        positions = [pos] if isinstance(pos, ContinuousPos) else list(pos)
        if not positions:
            return

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

        spread_cells=spread* self.resolution
        weights= self.kernel_matrix(kernel_type=kernel, radius= spread_cells)
        offsets=make_offsets(spread)

        deltas = np.stack(offsets, axis=-1) / self.resolution
        dist2 = np.sum(deltas * deltas, axis=-1)

        # Apply splat for each position
        for p, v in zip(positions, values):
            idx=global_indices_from_center(p, offsets, torus)

            if torus:
                idx_tuple = tuple(idx[d].astype(int) for d in range(self.ndims))
                incoming = float(v) * weights
                apply_mode_at(idx_tuple, incoming, mode=mode, alpha=alpha)
            else:
                in_bounds = np.ones_like(dist2, dtype=bool)
                for d in range(self.ndims):
                    in_bounds &= (
                        (idx[d] >= 0) & (idx[d] < self.shape[d])
                    )
                clipped_idxs = [
                    idx[d][in_bounds].astype(int)
                    for d in range(self.ndims)
                ]
                clipped_idxs_tuple=tuple(clipped_idxs)
                incoming = (v * weights)[in_bounds]
                apply_mode_at(clipped_idxs_tuple, incoming, mode=mode, alpha=alpha)


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

        return weights
    
    
    def get_neighborhood_mask( 
            self, 
            center_pos: ContinuousPos, 
            radius: float, 
            torus: bool = False, 
            ) -> np.ndarray: 
            
            if radius < 0: raise ValueError("radius must be >= 0") 

            mask = np.zeros(self.shape, dtype=bool) 
            
            offsets=make_offsets(radius)
            radius2 = float(radius) ** 2 

            deltas_cont = np.stack(offsets, axis=-1) / self.resolution # shape (side, side, ndims) 
            dist2 = np.sum(deltas_cont**2, axis=-1) # x**2+ y**2 
            local_inside = dist2 <= radius2 # local mask onto global mask 

            target_indices=global_indices_from_center(center_pos, offsets, torus)
            
            mask[target_indices] = local_inside 

            return mask

    

class HasPropertyLayers:
    
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
        space_bounds = np.asanyarray(self.dimensions, dtype=float)
        if space_bounds.shape != layer.bounds.shape or not np.allclose(space_bounds, layer.bounds):
            raise ValueError("Layer bounds must match the space bounds.")
        
        if layer.name in self._property_layers:
            raise ValueError(f"Property layer {layer.name} already exists.")

        self._property_layers[layer.name] = layer

    def remove_property_layer(self, name: str) -> None:
        if name not in self._properties:
            raise KeyError(f"No property named '{name}'.")
        
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

    
    def __getattr__(self, name: str) -> Any:
        try:
            return self._property_layers[name]
        except KeyError as e:
            raise AttributeError(
                f"'{type(self).__name__}' object has no property layer called '{name}'"
            ) from e