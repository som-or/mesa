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
    
    @property
    def ndim(self) -> int:
        return self._data.ndim
    
    @property
    def size(self) -> np.ndarray:
        mins=self.bounds[:, 0]
        maxs = self.bounds[:, 1]
        return maxs - mins
    @property
    def shape(self) -> tuple:
        return tuple(int(math.ceil(s * self.resolution)) for s in self.size)

    
    @data.setter
    def data(self, value) -> None:
        self._data[:] = value

    def __init__(
        self,
        name: str,
        bounds: Any,
        resolution: float = 1.0,
        default_value: Any = 0.0,
        dtype: type | np.dtype = float,
        torus: bool =False
    ) -> None:
        if resolution <= 0:
            raise ValueError("resolution(cells per unit) must be > 0.")
        

        self.name = name
        self.torus=torus
        self.bounds = np.asanyarray(bounds, dtype=float)
        if not(self.bounds.shape==(2,) or self.bounds.shape==(2,2)):
            raise ValueError("bounds must be either[width, height] or [[xmin, xmax], [ymin, ymax]]")
        
        if self.bounds.shape==(2,):
            self.bounds = np.vstack(([0, 0], self.bounds)).T


        # self.ndim= int(self.bounds.shape[0])
        self.resolution = float(resolution)

        mins = self.bounds[:, 0]
        maxs = self.bounds[:, 1]
        if not np.all(maxs > mins):
            raise ValueError("Each bounds row must satisfy max > min.")

        # Raster shape: ceil(size * cells_per_unit)
        # self.shape: tuple[int, ...] = tuple(int(math.ceil(s * self.resolution)) for s in self.size)
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
        self.default_value= default_value
        self._data = np.full(self.shape, default_value, dtype=dtype)

    def __getitem__(self, idx):
        return self._data[idx]

    def __setitem__(self, idx, value):
        self._data[idx] = value
    
    def __array__(self, dtype=None):
        return np.asarray(self._data, dtype=dtype)
        #example use case
        # np.sum(space.pheromone)
        # np.mean(space.pheromone)
   
    @classmethod
    def from_data(
        cls,
        name: str,
        data: np.ndarray,
        *,
        bounds: Any,
        resolution: float | None = None,
        torus: bool =False
    ) -> "PropertyLayer":
        
        bounds_arr = np.asanyarray(bounds, dtype=float)
        if not(bounds_arr.shape==(2,) or bounds_arr.shape==(2,2)):
            raise ValueError("bounds must be either[width, height] or [[xmin, xmax], [ymin, ymax]]")
        
        if bounds_arr.shape==(2,):
            bounds_arr = np.vstack(([0, 0], bounds_arr)).T

        size = bounds_arr[:, 1] - bounds_arr[:, 0]
        if data.ndim != bounds_arr.shape[0]:
            raise ValueError("data.ndim must match number of dimensions in bounds.")

        inferred = np.array(data.shape, dtype=float) / size

        if not np.allclose(inferred, inferred[0]):
            raise ValueError(
                f"Cannot infer a single scalar resolution: inferred per-axis={inferred}. "
                "Pass an explicit per-axis resolution (if you support it) or fix bounds/data."
            )

        inferred_res = float(inferred[0])

        if resolution is not None:
            if not np.isclose(float(resolution), inferred_res):
                raise ValueError(
                    f"resolution={resolution} disagrees with inferred resolution={inferred_res} "
                    f"from data.shape={data.shape} and bounds size={size}."
                )
            res = float(resolution)
        else:
            res = inferred_res
            
        layer = cls(
            name=name,
            bounds=bounds_arr,
            resolution=float(res),
            default_value=0,
            dtype=data.dtype.type,
            torus=torus
        )
        if layer.shape != tuple(data.shape):
            raise ValueError(
                f"data.shape {data.shape} does not match expected shape {layer.shape} "
            )
        
        layer.data = data.copy()
        return layer

    def set_bounds(self, bounds: Any) -> None:
        bounds = np.asanyarray(bounds, dtype=float)
        if bounds.shape == (2,):
            bounds = np.vstack(([0, 0], bounds)).T
        if bounds.shape != (self.ndim, 2):
            raise ValueError("bounds shape mismatch")

        # Must keep same size to avoid changing shape/data meaning
        new_size = bounds[:, 1] - bounds[:, 0]
        if not np.allclose(new_size, self.size):
            raise ValueError("New bounds must have same size as existing bounds")

        self.bounds = bounds
    
    def pos_to_index(self, pos: ContinuousPos,) -> Index:

        p = np.asanyarray(pos, dtype=float)
        if p.shape != (self.ndim,):
            raise ValueError(f"pos must have length {self.ndim}.")

        mins = self.bounds[:, 0]
        raw = np.floor((p - mins) * self.resolution).astype(int)

        if self.torus:
            raw = raw % np.array(self.shape)
        else:
            if np.any(raw < 0) or np.any(raw >= np.array(self.shape)):
                raise IndexError(...)
        return tuple(raw.astype(int))
        
    def index_to_pos(self, index: Sequence[int], *, center: bool = True) -> tuple[float, ...]:
        """
        If center=True, returns the center of the raster cell; otherwise returns cells upper left corner.
        """
        idx = np.asanyarray(index, dtype=float)
        if idx.shape != (self.ndim,):
            raise ValueError(f"index must have length {self.ndim}.")

        mins = self.bounds[:, 0]
        offset = 0.5 if center else 0.0
        pos = mins + (idx + offset) / self.resolution
        return tuple(float(x) for x in pos)
    

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
        
        positions = np.array(pos)
        if not positions.size:
            return
        if positions.ndim == 1:
            if positions.shape[0] != self.ndim:
                raise ValueError("pos dimension mismatch")
            positions = positions.reshape(1, self.ndim)
        elif positions.shape[1] != self.ndim:
            raise ValueError("pos dimension mismatch")

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

        
        self.apply_mode_at(idx_tuple, values, mode=mode, alpha=alpha)

        return 

    
    def deposit_splat(
    self,
    pos: Sequence["ContinuousPos"] | "ContinuousPos",
    value: Callable[["ContinuousPos"], float] | Sequence[float] | int | float,
    mode: str = "set",
    alpha: float | None = None,
    kernel: str = "gaussian",
    spread: float = 1
) -> None:
        """
        Deposit values onto the grid with local spatial spreading (splat).

        kernel
            One of {"gaussian", "linear", "epanechnikov"}.
        spread
            Radius.
        """
        torus=self.torus
        if spread < 0:
            raise ValueError("spread must be >= 0")

        positions = np.array(pos, dtype=float)
        if not positions.size:
            return
        if positions.ndim == 1:
            if positions.shape[0] != self.ndim:
                raise ValueError("pos dimension mismatch")
            positions = positions.reshape(1, self.ndim)
        elif positions.shape[1] != self.ndim:
            raise ValueError("pos dimension mismatch")

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
            return

        weights= self.kernel_matrix(kernel_type=kernel, radius= spread)
        offsets=self.make_offsets(spread)
        

        # deltas = np.stack(offsets, axis=-1) / self.resolution
        # dist2 = np.sum(deltas * deltas, axis=-1)

        # Apply splat for each position
        for p, v in zip(positions, values):

            idx_tuple, bound_mask=self.global_indices_from_center(p, offsets, torus)
            incoming = float(v) * weights

            self.apply_mode_at(
                idx_tuple=tuple(idx[bound_mask] for idx in idx_tuple), 
                incoming=incoming[bound_mask], 
                mode=mode, 
                alpha=alpha
            )


    def kernel_matrix(self, kernel_type: str, radius: float):
        radius_cell = int(math.ceil(radius * self.resolution)) 
        side = 2 * radius_cell + 1
        offsets = np.indices((side,) * self.ndim, dtype=int)
        for d in range(self.ndim):
            offsets[d] -= radius_cell

        deltas = np.stack(offsets, axis=-1) / self.resolution
        dist2 = np.sum(deltas * deltas, axis=-1)

        if kernel_type == "gaussian":
            sigma = radius_cell / self.resolution if radius_cell> 0 else 1.0
            weights = np.exp(-dist2 / (2 * sigma * sigma))
            #to make gaussian splat circular as well: need to truncate the values outside circle 
            weights[dist2 > radius* radius] = 0.0
        elif kernel_type == "linear":
            d = np.sqrt(dist2)
            weights = np.clip(1 - d / radius, 0.0, 1.0)
        elif kernel_type == "epanechnikov":
            weights = np.clip(1 - dist2 / (radius * radius), 0.0, 1.0)
        else:
            raise ValueError(f"Unknown kernel: {kernel_type}")

        maximum = weights.max()
        if maximum > 0:
            weights = weights / maximum

        return weights
    
    
    def get_neighborhood_mask( 
        self, 
        center_pos: ContinuousPos, 
        radius: float, 
        include_center: bool = True
        ) -> np.ndarray: 

        torus=self.torus
        if radius < 0: raise ValueError("radius must be >= 0") 

        mask = np.zeros(self.shape, dtype=bool) 
        
        offsets=self.make_offsets(radius)
        radius2 = float(radius) ** 2 

        deltas_cont = np.stack(offsets, axis=-1) / self.resolution # shape (side, side, ndims) 
        dist2 = np.sum(deltas_cont**2, axis=-1) # x**2+ y**2 
        local_inside = dist2 <= radius2 # local mask onto global mask 

        target_indices, bound_mask=self.global_indices_from_center(center_pos, offsets, torus)
        
        mask[target_indices] = local_inside & bound_mask
        if not include_center:
            mask[self.pos_to_index(center_pos)]= False

        return mask
    
    def decay(self,T: int, type: str= "linear", k: float | None= None):
        """
        type:
            -"exponential" decay
            -"linear" decay
        T: total time step the value dies in
        """
        if type == "linear":
            np.subtract(self._data, (self._data - self.default_value) / T, out=self._data)
        elif type == "exponential":
            if k==None: raise ValueError("need decay rate: k for exponential decay")
            np.multiply(self._data, np.exp(-k / T), out=self._data)
        else:
            raise ValueError("unknown decay type, choose from: exponential or linear")

        np.maximum(self._data, self.default_value, out=self._data)
    
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

            # sequential-important if indices repeat)
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
        offsets = np.indices((side,) * self.ndim, dtype=int)
        return offsets - radius_cells 

    def global_indices_from_center(
        self,
        center_pos: ContinuousPos,
        offsets: np.ndarray,
        torus: bool,
    ) -> tuple[np.ndarray, ...]:
        
        center_idx = self.pos_to_index(center_pos)
        valid_mask=None

        target_indices = [] 
        for d, off in enumerate(offsets): 
            idx = center_idx[d] + off 
            if torus: 
                idx = idx % self.shape[d] 
            else:
                dim_valid = (idx >= 0) & (idx < self.shape[d])
                valid_mask = dim_valid if valid_mask is None else (valid_mask & dim_valid)
          
            target_indices.append(idx.astype(int))

        if self.torus:
            valid_mask = np.ones_like(target_indices[0], dtype=bool)

        
        return tuple(target_indices), valid_mask

    

class HasPropertyLayers:
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._property: dict[str, PropertyLayer] = {}

    def create_property(
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
            torus=self.torus
        )
        self._attach_property(layer)
        return layer
    def add_property(self, name:str, array:np.ndarray):
        layer=PropertyLayer.from_data(
            name=name,
            data=array,
            bounds=self.dimensions,
            resolution=None,
            torus=self.torus
        )
        self._attach_property(layer)
        return layer

    def _attach_property(self, layer: PropertyLayer) -> None:
        space_size = np.asanyarray(self.size, dtype=float)
        if space_size.shape != layer.size.shape or not np.allclose(space_size, layer.size):
            raise ValueError("Layer bounds size must match the space bounds size.")
        
        if layer.name in self._property:
            raise ValueError(f"Property layer {layer.name} already exists.")

        layer.set_bounds(self.dimensions)
        layer.torus= self.torus
        self._property[layer.name] = layer

    def remove_property(self, name: str) -> None:
        if name not in self._property:
            raise KeyError(f"No property named '{name}'.")
        
        del self._property[name]
    
    def __getattr__(self, name: str) -> Any:
        layers = self.__dict__.get("_property")
    
        if layers is not None and name in layers:
            return layers[name]
        
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute or property layer '{name}'"
        )
