"""2D shape generators returning sets of (x, z) coordinates.

All shapes are centered at origin (0, 0). Use transforms.translate() to position.
"""

from __future__ import annotations
import math
from typing import Set, Tuple

Coord2D = Tuple[int, int]


def filled_circle(radius: float) -> Set[Coord2D]:
    """Generate all (x, z) points inside a circle of given radius.

    Uses the midpoint circle algorithm principle: a point is inside
    if x² + z² <= r².
    """
    r = radius
    r_sq = r * r
    points = set()
    ri = math.ceil(r)
    for x in range(-ri, ri + 1):
        for z in range(-ri, ri + 1):
            if x * x + z * z <= r_sq:
                points.add((x, z))
    return points


def circle_outline(radius: float, thickness: int = 1) -> Set[Coord2D]:
    """Generate the outline (ring edge) of a circle.

    Points that are inside radius but outside (radius - thickness).
    """
    outer = filled_circle(radius)
    if thickness >= radius:
        return outer
    inner = filled_circle(radius - thickness)
    return outer - inner


def circle_ring(outer_radius: float, inner_radius: float) -> Set[Coord2D]:
    """Generate a ring (annulus) between two radii."""
    return filled_circle(outer_radius) - filled_circle(inner_radius)


def filled_rectangle(width: int, depth: int) -> Set[Coord2D]:
    """Generate all (x, z) points in a rectangle centered at origin.

    Width is along X axis, depth along Z axis.
    """
    hw = width // 2
    hd = depth // 2
    return {(x, z) for x in range(-hw, width - hw) for z in range(-hd, depth - hd)}


def rectangle_shell(width: int, depth: int, thickness: int = 1) -> Set[Coord2D]:
    """Generate hollow rectangle walls of given thickness."""
    outer = filled_rectangle(width, depth)
    if thickness * 2 >= min(width, depth):
        return outer
    inner = filled_rectangle(width - thickness * 2, depth - thickness * 2)
    return outer - inner


def filled_diamond(half_width: int) -> Set[Coord2D]:
    """Generate a filled diamond (45° rotated square).

    The diamond extends half_width blocks in each cardinal direction.
    A point (x, z) is inside if |x| + |z| <= half_width.
    """
    return {
        (x, z)
        for x in range(-half_width, half_width + 1)
        for z in range(-half_width, half_width + 1)
        if abs(x) + abs(z) <= half_width
    }


def diamond_shell(half_width: int, thickness: int = 1) -> Set[Coord2D]:
    """Generate hollow diamond walls."""
    outer = filled_diamond(half_width)
    if thickness >= half_width:
        return outer
    inner = filled_diamond(half_width - thickness)
    return outer - inner


def octagon(diameter: int) -> Set[Coord2D]:
    """Generate a filled octagon by cutting corners off a square.

    Good for approximating circular towers (Hotel New York).
    Corner cut size is ~diameter/4 rounded.
    """
    half = diameter // 2
    cut = max(1, round(diameter / 4))
    points = set()
    for x in range(-half, diameter - half):
        for z in range(-half, diameter - half):
            # Offset from center
            cx = abs(x - (diameter // 2 - half))
            cz = abs(z - (diameter // 2 - half))
            # Inside square
            if 0 <= x + half < diameter and 0 <= z + half < diameter:
                # Cut corners: distance from corner must be > cut
                in_nw = (x + half) + (z + half) >= cut
                in_ne = (diameter - 1 - (x + half)) + (z + half) >= cut
                in_sw = (x + half) + (diameter - 1 - (z + half)) >= cut
                in_se = (diameter - 1 - (x + half)) + (diameter - 1 - (z + half)) >= cut
                if in_nw and in_ne and in_sw and in_se:
                    points.add((x, z))
    return points


def hexagon(width: int) -> Set[Coord2D]:
    """Generate a filled hexagon approximation.

    Flat-top orientation, used for Cube Houses pylons.
    """
    half = width // 2
    points = set()
    for z in range(-half, half + 1):
        # At each z row, the x range shrinks toward top/bottom
        # Hexagon: width at z is width - |z|
        x_range = half - abs(z) // 2
        for x in range(-x_range, x_range + 1):
            points.add((x, z))
    return points


def variable_rectangle(width: int, depth: int, corner_x: int = 0, corner_z: int = 0) -> Set[Coord2D]:
    """Rectangle anchored at a specific corner instead of centered.

    Useful for towers and shifted volumes.
    """
    return {
        (corner_x + x, corner_z + z)
        for x in range(width)
        for z in range(depth)
    }
