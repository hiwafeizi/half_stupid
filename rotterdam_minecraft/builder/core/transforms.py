"""Spatial transformations on sets of 3D block coordinates."""

from __future__ import annotations
from typing import Set, Tuple, Dict

Coord3D = Tuple[int, int, int]


def translate(
    blocks: Set[Coord3D],
    dx: int = 0,
    dy: int = 0,
    dz: int = 0,
) -> Set[Coord3D]:
    """Move all blocks by (dx, dy, dz)."""
    return {(x + dx, y + dy, z + dz) for x, y, z in blocks}


def translate_dict(
    blocks: Dict[Coord3D, str],
    dx: int = 0,
    dy: int = 0,
    dz: int = 0,
) -> Dict[Coord3D, str]:
    """Move all blocks in a {coord: block_type} dict."""
    return {(x + dx, y + dy, z + dz): b for (x, y, z), b in blocks.items()}


def mirror_x(blocks: Set[Coord3D], center_x: int = 0) -> Set[Coord3D]:
    """Mirror blocks across the X axis (reflect x coordinate)."""
    return {(2 * center_x - x, y, z) for x, y, z in blocks}


def mirror_z(blocks: Set[Coord3D], center_z: int = 0) -> Set[Coord3D]:
    """Mirror blocks across the Z axis (reflect z coordinate)."""
    return {(x, y, 2 * center_z - z) for x, y, z in blocks}


def mirror_y(blocks: Set[Coord3D], center_y: int = 0) -> Set[Coord3D]:
    """Mirror blocks across the Y axis (reflect y coordinate)."""
    return {(x, 2 * center_y - y, z) for x, y, z in blocks}


def rotate_90(
    blocks: Set[Coord3D],
    center_x: int = 0,
    center_z: int = 0,
    times: int = 1,
) -> Set[Coord3D]:
    """Rotate blocks 90° clockwise around Y axis, `times` times.

    Rotation center at (center_x, ?, center_z).
    """
    result = set(blocks)
    for _ in range(times % 4):
        result = {
            (center_x + (z - center_z), y, center_z - (x - center_x))
            for x, y, z in result
        }
    return result


def scale_2d_to_3d(
    profile_2d: Set[Tuple[int, int]],
    y: int,
    plane: str = "xz",
) -> Set[Coord3D]:
    """Convert a 2D profile to 3D points at a fixed height.

    Args:
        profile_2d: Set of (a, b) 2D points.
        y: The Y coordinate for the layer.
        plane: "xz" -> (a, y, b), "xy" -> (a, y_val, b) where y_val = b and z = y.
    """
    if plane == "xz":
        return {(a, y, b) for a, b in profile_2d}
    elif plane == "xy":
        return {(a, b, y) for a, b in profile_2d}
    else:
        raise ValueError(f"Unknown plane: {plane}")


def offset_blocks(
    blocks: Set[Coord3D],
    y_ranges: list,
) -> Set[Coord3D]:
    """Apply different XZ offsets to blocks based on Y ranges.

    Used for De Rotterdam tower shifts.

    Args:
        y_ranges: List of {"y_min": int, "y_max": int, "dx": int, "dz": int}
    """
    result = set()
    for x, y, z in blocks:
        dx, dz = 0, 0
        for r in y_ranges:
            if r["y_min"] <= y < r["y_max"]:
                dx = r.get("dx", 0)
                dz = r.get("dz", 0)
                break
        result.add((x + dx, y + dy, z + dz) if False else (x + dx, y, z + dz))
    return result
