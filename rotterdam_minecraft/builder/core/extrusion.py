"""Extrude 2D profiles into 3D volumes.

The primary strategy for generating most buildings:
1. Define a 2D cross-section (circle, rectangle, arch profile)
2. Extrude it along an axis (Y for vertical, X or Z for horizontal)
"""

from __future__ import annotations
from typing import Set, Tuple, Callable, Optional

Coord2D = Tuple[int, int]
Coord3D = Tuple[int, int, int]


def extrude_constant(
    profile: Set[Coord2D],
    axis: str,
    length: int,
    start: int = 0,
) -> Set[Coord3D]:
    """Extrude a 2D profile along an axis for a fixed length.

    Args:
        profile: Set of (a, b) points in the 2D cross-section.
        axis: "y" (vertical), "x", or "z" (horizontal extrusion).
        length: How many blocks to extrude.
        start: Starting coordinate along the extrusion axis.

    Returns:
        Set of (x, y, z) points.

    Examples:
        - Markthal: arch profile extruded along Z for 120 blocks.
        - Van Nelle: factory cross-section extruded along X.
        - Euromast: circle extruded along Y (same as extrude_cylinder).
    """
    points = set()
    for i in range(start, start + length):
        for a, b in profile:
            if axis == "y":
                points.add((a, i, b))     # profile is (x, z), extruded along y
            elif axis == "x":
                points.add((i, a, b))     # profile is (y, z), extruded along x
            elif axis == "z":
                points.add((a, b, i))     # profile is (x, y), extruded along z
    return points


def extrude_variable(
    profile_fn: Callable[[int], Set[Coord2D]],
    axis: str,
    length: int,
    start: int = 0,
) -> Set[Coord3D]:
    """Extrude with a profile that changes at each layer.

    Args:
        profile_fn: Function taking layer index -> set of (a, b) 2D points.
        axis: "y", "x", or "z".
        length: Total extrusion length.
        start: Starting coordinate.

    Returns:
        Set of (x, y, z) points.

    Examples:
        - Depot Boijmans: circle_ring(d(y)/2, d(y)/2 - wall) at each Y.
        - Witte Huis mansard: rectangle shrinking per Y layer.
        - Cube Houses: diamond expanding then contracting per Y layer.
    """
    points = set()
    for i in range(start, start + length):
        profile = profile_fn(i - start)
        for a, b in profile:
            if axis == "y":
                points.add((a, i, b))
            elif axis == "x":
                points.add((i, a, b))
            elif axis == "z":
                points.add((a, b, i))
    return points


def extrude_cylinder(
    radius: float,
    height: int,
    start_y: int = 0,
    hollow: bool = False,
    wall_thickness: int = 1,
) -> Set[Coord3D]:
    """Extrude a circle along Y to create a cylinder.

    Convenience wrapper for extrude_constant with a circle profile.

    Args:
        radius: Circle radius.
        height: Height in blocks.
        start_y: Y coordinate of bottom.
        hollow: If True, only the wall ring (not filled).
        wall_thickness: Wall thickness when hollow.
    """
    from .shapes import filled_circle, circle_ring

    if hollow:
        profile = circle_ring(radius, radius - wall_thickness)
    else:
        profile = filled_circle(radius)

    return extrude_constant(profile, "y", height, start=start_y)


def extrude_ring(
    outer_radius: float,
    inner_radius: float,
    height: int,
    start_y: int = 0,
) -> Set[Coord3D]:
    """Extrude a ring (annulus) along Y.

    Convenience wrapper combining circle_ring with extrude_constant.
    """
    from .shapes import circle_ring

    profile = circle_ring(outer_radius, inner_radius)
    return extrude_constant(profile, "y", height, start=start_y)


def extrude_box(
    width: int,
    height: int,
    depth: int,
    start_x: int = 0,
    start_y: int = 0,
    start_z: int = 0,
    hollow: bool = False,
    wall_thickness: int = 1,
) -> Set[Coord3D]:
    """Generate a rectangular box (filled or hollow shell).

    Simpler than using extrude_constant for basic rectangular volumes.

    Args:
        width: Size along X.
        height: Size along Y.
        depth: Size along Z.
        start_x/y/z: Origin corner.
        hollow: If True, only walls (no interior fill).
        wall_thickness: Wall thickness when hollow.
    """
    points = set()
    for x in range(start_x, start_x + width):
        for y in range(start_y, start_y + height):
            for z in range(start_z, start_z + depth):
                if hollow:
                    # Check if on any wall face
                    on_x_wall = x < start_x + wall_thickness or x >= start_x + width - wall_thickness
                    on_y_wall = y < start_y + wall_thickness or y >= start_y + height - wall_thickness
                    on_z_wall = z < start_z + wall_thickness or z >= start_z + depth - wall_thickness
                    if on_x_wall or on_y_wall or on_z_wall:
                        points.add((x, y, z))
                else:
                    points.add((x, y, z))
    return points


def extrude_arch(
    layer_data: list,
    length: int,
    axis: str = "z",
    start: int = 0,
    wall_only: bool = True,
) -> Set[Coord3D]:
    """Extrude a horseshoe/arch profile along an axis.

    Designed for the Markthal: takes layer-by-layer width data and
    extrudes the same cross-section for `length` blocks.

    Args:
        layer_data: List of dicts with "height", "exterior_width", "interior_width".
        length: Extrusion length.
        axis: Extrusion axis ("z" for Markthal).
        start: Start coordinate.
        wall_only: If True, only the shell (exterior - interior). If False, solid.
    """
    from .shapes import filled_rectangle

    points = set()
    for layer in layer_data:
        y = layer["height"]
        ext_w = layer["exterior_width"]
        int_w = layer.get("interior_width", 0)

        if wall_only and int_w > 0:
            # Shell: exterior minus interior
            ext_profile = filled_rectangle(ext_w, 1)  # 1-deep slice
            int_profile = filled_rectangle(int_w, 1)
            profile = ext_profile - int_profile
        else:
            profile = filled_rectangle(ext_w, 1)

        for i in range(start, start + length):
            for x, _ in profile:
                if axis == "z":
                    points.add((x, y, i))
                elif axis == "x":
                    points.add((i, y, x))
    return points
