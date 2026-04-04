"""Curve generators for arcs, parabolas, diagonal lines, and profiles."""

from __future__ import annotations
import math
from typing import List, Set, Tuple, Callable

Coord3D = Tuple[int, int, int]
Coord2D = Tuple[int, int]


def bresenham_line_3d(
    x0: int, y0: int, z0: int,
    x1: int, y1: int, z1: int,
) -> List[Coord3D]:
    """3D Bresenham line algorithm. Returns list of (x, y, z) points.

    Used for cables (Erasmus Bridge) and conveyor bridges (Van Nelle).
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    sz = 1 if z1 > z0 else -1

    # Driving axis is the one with the largest delta
    if dx >= dy and dx >= dz:
        ey = 2 * dy - dx
        ez = 2 * dz - dx
        x, y, z = x0, y0, z0
        for _ in range(dx + 1):
            points.append((x, y, z))
            if ey > 0:
                y += sy
                ey -= 2 * dx
            if ez > 0:
                z += sz
                ez -= 2 * dx
            ey += 2 * dy
            ez += 2 * dz
            x += sx
    elif dy >= dx and dy >= dz:
        ex = 2 * dx - dy
        ez = 2 * dz - dy
        x, y, z = x0, y0, z0
        for _ in range(dy + 1):
            points.append((x, y, z))
            if ex > 0:
                x += sx
                ex -= 2 * dy
            if ez > 0:
                z += sz
                ez -= 2 * dy
            ex += 2 * dx
            ez += 2 * dz
            y += sy
    else:
        ex = 2 * dx - dz
        ey = 2 * dy - dz
        x, y, z = x0, y0, z0
        for _ in range(dz + 1):
            points.append((x, y, z))
            if ex > 0:
                x += sx
                ex -= 2 * dz
            if ey > 0:
                y += sy
                ey -= 2 * dz
            ex += 2 * dx
            ey += 2 * dy
            z += sz

    return points


def arc_points(
    radius: float,
    start_angle: float,
    end_angle: float,
    axis: str = "xz",
) -> Set[Coord2D]:
    """Generate 2D points along an arc segment.

    Angles in degrees. 0° = positive x-axis, 90° = positive z-axis.
    Used for Van Nelle curved office building.

    Args:
        radius: Arc radius.
        start_angle: Start angle in degrees.
        end_angle: End angle in degrees.
        axis: Which plane ("xz" or "xy").
    """
    points = set()
    # Sample at sub-block resolution for smooth arcs
    steps = max(int(abs(end_angle - start_angle) * radius * 0.1), 100)
    for i in range(steps + 1):
        t = math.radians(start_angle + (end_angle - start_angle) * i / steps)
        x = round(radius * math.cos(t))
        z = round(radius * math.sin(t))
        points.add((x, z))
    return points


def arc_filled(
    outer_radius: float,
    inner_radius: float,
    start_angle: float,
    end_angle: float,
) -> Set[Coord2D]:
    """Generate filled arc sector (pie slice with hole).

    Returns (x, z) points between inner and outer radius within the angle range.
    """
    points = set()
    ri = math.ceil(outer_radius)
    start_rad = math.radians(start_angle)
    end_rad = math.radians(end_angle)
    if start_rad > end_rad:
        start_rad, end_rad = end_rad, start_rad

    for x in range(-ri, ri + 1):
        for z in range(-ri, ri + 1):
            dist_sq = x * x + z * z
            if inner_radius * inner_radius <= dist_sq <= outer_radius * outer_radius:
                angle = math.atan2(z, x)
                if angle < 0:
                    angle += 2 * math.pi
                if start_rad <= angle <= end_rad:
                    points.add((x, z))
    return points


def parabolic_diameter(
    base_diameter: float,
    top_diameter: float,
    height: int,
    exponent: float = 1.6,
) -> List[float]:
    """Generate diameter at each height layer for a parabolic profile.

    Used by Depot Boijmans: d(h) = base + (top - base) * (h/height)^exponent

    Returns list of diameters, index = height layer (0 to height inclusive).
    """
    diff = top_diameter - base_diameter
    return [
        base_diameter + diff * (h / height) ** exponent
        for h in range(height + 1)
    ]


def piecewise_slope(
    segments: List[dict],
) -> List[Coord3D]:
    """Generate 3D points along a piecewise linear slope.

    Each segment: {"start": (x,y,z), "end": (x,y,z)}
    Returns all Bresenham points along all segments concatenated.

    Used for Erasmus Bridge pylon kink, Rotterdam Centraal canopy profile.
    """
    all_points = []
    for seg in segments:
        sx, sy, sz = seg["start"]
        ex, ey, ez = seg["end"]
        pts = bresenham_line_3d(sx, sy, sz, ex, ey, ez)
        # Avoid duplicate at segment joints
        if all_points and pts and pts[0] == all_points[-1]:
            pts = pts[1:]
        all_points.extend(pts)
    return all_points


def horseshoe_profile(
    layer_data: List[dict],
) -> List[dict]:
    """Parse a horseshoe arch profile from layer-by-layer JSON data.

    Each entry in layer_data should have:
        {"height": int, "interior_width": int, "exterior_width": int}

    Returns the same list, validated and ready for extrusion.
    Used by Markthal.
    """
    validated = []
    for layer in layer_data:
        validated.append({
            "height": int(layer["height"]),
            "interior_width": int(layer.get("interior_width", 0)),
            "exterior_width": int(layer["exterior_width"]),
        })
    return validated


def slope_profile(
    width: int,
    peak_height: int,
    peak_x_offset: int,
    left_slope: float,
    right_slope: float,
) -> List[Tuple[int, int, int]]:
    """Generate an asymmetric peaked roof profile.

    Returns list of (x, y, block_present) for one cross-section row.
    Used by Rotterdam Centraal canopy.

    Args:
        width: Total width in blocks.
        peak_height: Height at the peak.
        peak_x_offset: X position of peak from left edge.
        left_slope: Rise/run ratio for left side (steeper = higher value).
        right_slope: Rise/run ratio for right side.
    """
    profile = []
    for x in range(width):
        if x <= peak_x_offset:
            # Left side: rise from 0 to peak_height
            dist_from_edge = x
            height = min(peak_height, round(dist_from_edge * left_slope))
        else:
            # Right side: fall from peak_height to 0
            dist_from_peak = x - peak_x_offset
            height = max(0, peak_height - round(dist_from_peak * right_slope))
        profile.append((x, height))
    return profile
