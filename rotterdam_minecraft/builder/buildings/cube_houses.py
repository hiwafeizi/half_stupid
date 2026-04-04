"""Cube Houses (Kubuswoningen) builder — 38+2 tilted cubes.

Each cube: 12 blocks tall diamond on a 10-block hexagonal pylon = 22 total.
38 small cubes + 2 super-cubes arranged in a continuous row.
"""

from ..engine.world import World
from ..engine.palette import CUBE_HOUSES, Palette
from ..core.shapes import filled_diamond, diamond_shell, hexagon
from ..patterns.repeater import array_along_path


def _build_single_cube(p, is_super: bool = False) -> dict:
    """Build one cube unit (pylon + tilted cube). Returns block dict centered at origin."""
    blocks = {}
    scale = 1.5 if is_super else 1.0
    pylon_h = round(10 * scale)
    cube_h = round(12 * scale)
    max_hw = round(6 * scale)  # Max diamond half-width

    # --- Pylon (hexagonal column) ---
    hex_w = round(4 * scale)
    hex_profile = hexagon(hex_w)
    for y in range(pylon_h):
        for x, z in hex_profile:
            blocks[(x, y, z)] = p["pylon"]

    # --- Tilted cube (diamond shape, expanding then contracting) ---
    cube_start_y = pylon_h

    for layer in range(cube_h):
        t = layer / (cube_h - 1) if cube_h > 1 else 0
        # Expand to max at midpoint, then contract
        if t <= 0.5:
            hw = max(1, round(max_hw * t * 2))
        else:
            hw = max(1, round(max_hw * (1 - t) * 2))

        y = cube_start_y + layer

        # Determine if this is a window layer
        is_window_layer = 0.2 < t < 0.8

        shell = diamond_shell(hw, thickness=1)
        for x, z in shell:
            if is_window_layer and abs(x) + abs(z) < hw:
                blocks[(x, y, z)] = p["windows"]
            else:
                blocks[(x, y, z)] = p["cube_walls"]

        # Interior floors at level transitions
        if layer in [round(cube_h * 0.25), round(cube_h * 0.5)]:
            interior = filled_diamond(max(1, hw - 1))
            for x, z in interior:
                blocks[(x, y, z)] = p["interior_floor"]

    return blocks


def build(world: World, palette: Palette = None, origin_x: int = 0, origin_z: int = 0) -> None:
    """Build Cube Houses at 1:1 scale (38 small + 2 super cubes)."""
    p = palette or CUBE_HOUSES

    # Build templates
    small_cube = _build_single_cube(p, is_super=False)
    super_cube = _build_single_cube(p, is_super=True)

    # Arrangement: cubes in a row along X axis
    # Small cubes spaced ~11 blocks apart (diamond diagonal is ~11)
    SMALL_SPACING = 11
    SUPER_SPACING = 16

    # Layout: super-cube, 19 small, gap, 19 small, super-cube
    positions = []

    # First super-cube
    positions.append(("super", origin_x - 19 * SMALL_SPACING // 2 - SUPER_SPACING, origin_z))

    # First row of 19 small cubes
    start_x = origin_x - 19 * SMALL_SPACING // 2
    for i in range(19):
        positions.append(("small", start_x + i * SMALL_SPACING, origin_z))

    # Second row of 19 small cubes (slight Z offset for the zigzag)
    for i in range(19):
        positions.append(("small", start_x + i * SMALL_SPACING, origin_z + 6))

    # Second super-cube
    positions.append(("super", start_x + 19 * SMALL_SPACING, origin_z + 6))

    # Place all cubes
    for cube_type, cx, cz in positions:
        template = super_cube if cube_type == "super" else small_cube
        for (x, y, z), block in template.items():
            world.set_block(cx + x, y, cz + z, block)

    # === Ground-level walkway/market area ===
    walkway_start = positions[0][1] - 10
    walkway_end = positions[-1][1] + 10
    for x in range(walkway_start, walkway_end):
        for z in range(origin_z - 8, origin_z + 14):
            world.set_block(x, 0, z, p["walkway_floor"])
