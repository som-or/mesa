"""
Test script for ContinuousSpace PropertyLayers (raster field overlay).

What this covers:
- HasPropertyLayers lifecycle: create/add/get/remove, getattr access
- PropertyLayer: pos_to_index, index_to_pos
- set_cells / modify_cells (ufunc + callable) / select_cells / aggregate
- neighborhood mask (torus + non-torus)
- diffuse
- space-level forwarding: get_value/set_value, set_property/modify_property, get_neighborhood_mask

Run:
    python test_continuous_property_layers.py
"""

from __future__ import annotations

import numpy as np

# ---- Paste / import your implementation here ----
from mesa.experimental.continuous_space.property_layer import PropertyLayer, HasPropertyLayers

# For convenience, assume the classes are already in scope:
# PropertyLayer, HasPropertyLayers


class DummyContinuousSpace(HasPropertyLayers):
    """A minimal ContinuousSpace-like host for HasPropertyLayers.

    Requirements used by the mixin:
    - self.dimensions : array-like shape (ndims, 2)
    - self.torus : bool
    """

    def __init__(self, dimensions, torus: bool = False):
        self.dimensions = np.asanyarray(dimensions, dtype=float)
        self.torus = bool(torus)
        super().__init__()


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    # -------------------------------------------------------------------------
    # 0) Create a dummy continuous space
    # -------------------------------------------------------------------------
    print_header("0) Create DummyContinuousSpace")
    bounds = np.array([[0.0, 10.0], [0.0, 5.0]])  # width=10, height=5
    space = DummyContinuousSpace(bounds, torus=False)
    print("Space bounds:\n", space.dimensions)
    print("Space torus:", space.torus)

    # -------------------------------------------------------------------------
    # 1) Create a property layer through the mixin
    # -------------------------------------------------------------------------
    print_header("1) create_property_layer + basic metadata")
    temp = space.create_property_layer("temperature", resolution=2.0, default_value=0.0, dtype=float)
    print("Layer name:", temp.name)
    print("Layer bounds:\n", temp.bounds)
    print("Layer resolution:", temp.resolution)
    print("Layer shape:", temp.shape)
    print("Data stats (min/max):", temp.data.min(), temp.data.max())

    # Also test __getattr__ access to layer
    assert space.temperature is temp
    print("Access via space.temperature OK")

    # -------------------------------------------------------------------------
    # 2) Coordinate mapping: pos_to_index / index_to_pos
    # -------------------------------------------------------------------------
    print_header("2) pos_to_index / index_to_pos")
    pos = (3.2, 4.9)
    idx = temp.pos_to_index(pos)
    back = temp.index_to_pos(idx, center=True)
    print("pos:", pos, "-> idx:", idx, "-> back(center):", back)

    # Edge clamping
    pos_outside = (-100.0, 999.0)
    idx_clamped = temp.pos_to_index(pos_outside, clamp=True)
    print("pos_outside:", pos_outside, "-> idx_clamped:", idx_clamped)

    # -------------------------------------------------------------------------
    # 3) set_cells: full overwrite and conditional overwrite
    # -------------------------------------------------------------------------
    print_header("3) set_cells")
    temp.set_cells(1.0)
    print("After set_cells(1.0): min/max =", temp.data.min(), temp.data.max())

    # Condition: set all cells > 0.5 to 2.0 (should set everything to 2.0 here)
    temp.set_cells(2.0, condition=lambda a: a > 0.5)
    print("After conditional set_cells(2.0, a>0.5): unique =", np.unique(temp.data))

    # -------------------------------------------------------------------------
    # 4) modify_cells: ufunc and callable, with mask
    # -------------------------------------------------------------------------
    print_header("4) modify_cells (ufunc + callable)")

    # Start fresh with a gradient so we have something interesting
    # Make data increase with x index
    x = np.arange(temp.shape[0])[:, None]
    temp.data[:] = x  # broadcast across y
    print("Gradient set. Sample row 0..5 at y=0:", temp.data[:6, 0])

    # Ufunc: add 10 to cells where value < 3
    temp.modify_cells(np.add, value=10, condition=lambda a: a < 3)
    print("After add 10 where a<3. Sample row 0..5:", temp.data[:6, 0])

    # Callable: clamp values to max 8 where values > 8
    def clamp_max(arr, vmax):
        return np.minimum(arr, vmax)

    temp.modify_cells(clamp_max, value=8, condition=lambda a: a > 8)
    print("After clamp to 8 where a>8. Sample row 0..10:", temp.data[:11, 0])

    # -------------------------------------------------------------------------
    # 5) select_cells + aggregate
    # -------------------------------------------------------------------------
    print_header("5) select_cells + aggregate")

    # Select cells equal to 8 (should be all high-x rows)
    coords = temp.select_cells(lambda a: a == 8, return_list=True)
    print("Number of coords where value==8:", len(coords))
    print("First 10 coords:", coords[:10])

    mask = temp.select_cells(lambda a: a == 8, return_list=False)
    print("Mask dtype/shape:", mask.dtype, mask.shape)

    # Aggregate
    total = temp.aggregate(np.sum)
    mean = temp.aggregate(np.mean)
    print("Aggregate sum:", float(total))
    print("Aggregate mean:", float(mean))

    # -------------------------------------------------------------------------
    # 6) Neighborhood mask (non-torus)
    # -------------------------------------------------------------------------
    print_header("6) get_neighborhood_mask (non-torus)")
    center = (5.0, 2.5)
    radius = 1.2
    neigh_mask = temp.get_neighborhood_mask(center, radius, torus=False)
    print("Neighborhood mask True count:", int(neigh_mask.sum()))
    print("Neighborhood mask shape:", neigh_mask.shape)

    # Use neighborhood mask to set a “hotspot”
    temp.set_cells(99.0, condition=lambda a, m=neigh_mask: m)
    print("After hotspot set via mask: max =", temp.data.max())

    # -------------------------------------------------------------------------
    # 7) Diffuse (global)
    # -------------------------------------------------------------------------
    print_header("7) diffuse (global Gaussian)")
    before_max = float(temp.data.max())
    temp.diffuse(distance=0.8, diffusion_rate=1.0, torus=False)
    after_max = float(temp.data.max())
    print("Max before diffuse:", before_max, "Max after diffuse:", after_max)
    print("Center sample (idx near center):", temp.data[temp.pos_to_index(center)])

    # -------------------------------------------------------------------------
    # 8) Space-level get_value / set_value and bulk forwarding
    # -------------------------------------------------------------------------
    print_header("8) Space-level convenience APIs")

    p = (1.1, 1.1)
    v_before = space.get_value("temperature", p)
    space.set_value("temperature", p, 123.0)
    v_after = space.get_value("temperature", p)
    print("get_value before:", v_before, "after set_value:", v_after)

    # Forwarding: set_property / modify_property
    # Set all cells < 1 to -5
    space.set_property("temperature", -5.0, condition=lambda a: a < 1.0)
    print("After set_property(-5 where a<1): min =", float(space.temperature.data.min()))

    # Multiply all cells by 0.5 where value > 10 (ufunc multiply)
    space.modify_property("temperature", np.multiply, value=0.5, condition=lambda a: a > 10.0)
    print("After modify_property(multiply 0.5 where a>10): max =", float(space.temperature.data.max()))

    # Space sugar neighborhood mask
    neigh2 = space.get_neighborhood_mask("temperature", center_pos=center, radius=1.2, torus=False)
    print("Space.get_neighborhood_mask True count:", int(neigh2.sum()))

    # -------------------------------------------------------------------------
    # 9) Torus neighborhood check (wrap)
    # -------------------------------------------------------------------------
    print_header("9) Torus neighborhood mask behavior (wrap check)")
    torus_space = DummyContinuousSpace(bounds, torus=True)
    torus_layer = torus_space.create_property_layer("pheromone", resolution=2.0, default_value=0.0)
    # Put a spike near the left edge
    torus_space.set_value("pheromone", (0.1, 2.5), 50.0)

    # Neighborhood centered near left edge should wrap if torus=True
    wrap_center = (0.1, 2.5)
    wrap_mask = torus_layer.get_neighborhood_mask(wrap_center, radius=1.0, torus=True)
    print("Wrap mask True count:", int(wrap_mask.sum()))

    # Demonstrate that some of the selected indices include the rightmost columns (wrap)
    wrap_coords = list(zip(*np.where(wrap_mask)))
    xs = [c[0] for c in wrap_coords]
    print("Min x index in wrap neighborhood:", min(xs), "Max x index:", max(xs))
    print("Layer x-size:", torus_layer.shape[0], "(expect some indices near end if wrapping occurred)")

    # -------------------------------------------------------------------------
    # 10) add_property_layer with explicit layer + from_data
    # -------------------------------------------------------------------------
    print_header("10) add_property_layer + from_data")

    # Make a separate layer and add it
    humidity = PropertyLayer("humidity", bounds=bounds, resolution=1.0, default_value=7.0, dtype=float)
    space.add_property_layer(humidity)
    print("Added humidity. Access via get_property_layer:", space.get_property_layer("humidity").name)
    print("Humidity unique:", np.unique(space.humidity.data))

    # from_data roundtrip
    data = np.zeros((int(np.ceil(10 * 1.0)), int(np.ceil(5 * 1.0))), dtype=np.float64)
    data[0, 0] = 42.0
    imported = PropertyLayer.from_data("imported", data, bounds=bounds, resolution=1.0)
    space.add_property_layer(imported)
    print("Imported layer shape:", imported.shape, "value(0,0):", imported.data[0, 0])

    # -------------------------------------------------------------------------
    # 11) remove_property_layer
    # -------------------------------------------------------------------------
    print_header("11) remove_property_layer")
    space.remove_property_layer("imported")
    print("Removed 'imported'. Remaining layers:", list(space._property_layers.keys()))

    # Done
    print_header("ALL TESTS COMPLETED")
    print("If you saw no exceptions, the core API is coherent.\n")


if __name__ == "__main__":
    # If you are running this as a standalone file, ensure PropertyLayer/HasPropertyLayers
    # are imported or pasted above.
    try:
        main()
    except NameError as e:
        raise SystemExit(
            "NameError: Did you forget to import or paste PropertyLayer/HasPropertyLayers above?\n"
            f"Original error: {e}"
        )