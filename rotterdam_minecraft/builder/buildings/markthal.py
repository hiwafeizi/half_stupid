"""Markthal builder — giant horseshoe arch with market inside.

40 blocks tall, 120 long, 70 wide. Key: arch profile + glass end walls + mural ceiling.
"""

from ..engine.world import World
from ..engine.palette import MARKTHAL, Palette
from ..core.shapes import filled_rectangle, rectangle_shell
from ..patterns.curtain_wall import cable_net_glass
from ..patterns.window_grid import window_grid
from ..patterns.floor_stack import floor_stack
import math
import random


def _arch_profile(y: int, ext_w: int = 70, wall_t: int = 12) -> tuple:
    """Calculate interior and exterior widths at height y.

    The Markthal arch: lower 55% is near-vertical walls,
    upper portion curves inward to meet at the crown.
    """
    if y <= 22:
        # Vertical section: walls are straight
        return ext_w, ext_w - wall_t * 2
    elif y <= 38:
        # Curving section: both narrow
        progress = (y - 22) / 16  # 0 to 1
        # Exterior narrows slightly
        ext = ext_w - int(progress ** 1.5 * 20)
        # Interior narrows faster
        interior_base = ext_w - wall_t * 2
        int_w = interior_base - int(progress ** 1.3 * (interior_base - 4))
        return max(4, ext), max(0, int_w)
    else:
        # Crown cap: solid
        ext = max(4, ext_w - int(0.9 * ext_w * ((y - 22) / 18) ** 1.5))
        return ext, 0


def build(world: World, palette: Palette = None, origin_x: int = 0, origin_z: int = 0) -> None:
    """Build the Markthal at 1:1 scale (120x70x40)."""
    p = palette or MARKTHAL

    LENGTH = 120   # Along Z axis
    WIDTH = 70     # Along X axis
    HEIGHT = 40
    WALL_T = 12   # Arch wall thickness

    half_w = WIDTH // 2
    half_l = LENGTH // 2

    # ===== 1. Arch shell (extruded cross-section along Z) =====
    for z in range(origin_z - half_l, origin_z + half_l):
        for y in range(HEIGHT):
            ext_w, int_w = _arch_profile(y, WIDTH, WALL_T)
            ext_half = ext_w // 2
            int_half = int_w // 2

            for x in range(-ext_half, ext_half):
                # Check if this is wall (not interior)
                if int_w <= 0 or abs(x) >= int_half:
                    world.set_block(origin_x + x, y, z, p["exterior"])

    # ===== 2. Exterior windows (apartment windows on the arch surface) =====
    # Windows appear on floors 3-11 of the apartment shell
    for z in range(origin_z - half_l + 2, origin_z + half_l - 2, 4):
        for y in range(8, 35, 4):  # Window rows
            ext_w, int_w = _arch_profile(y, WIDTH, WALL_T)
            ext_half = ext_w // 2
            # Place windows on outer faces
            for side in [-1, 1]:
                wx = origin_x + side * (ext_half - 1)
                world.set_block(wx, y, z, p["apartment_walls"])
                world.set_block(wx, y + 1, z, p["apartment_walls"])

    # ===== 3. Glass end walls (both open ends) =====
    for end_z in [origin_z - half_l, origin_z + half_l - 1]:
        # Glass wall filling the arch opening
        for y in range(HEIGHT):
            _, int_w = _arch_profile(y, WIDTH, WALL_T)
            if int_w > 0:
                glass = cable_net_glass(
                    width=int_w, height=1,
                    cable_spacing=5,
                    start_x=origin_x - int_w // 2,
                    start_y=y, start_z=end_z,
                    face="z",
                    glass_block=p["glass_ends"],
                    cable_block=p["cable_net"],
                )
                world.set_block_dict(glass)

    # ===== 4. Interior mural ceiling =====
    # Color zones on the interior ceiling surface
    mural_colors = [
        p["mural_warm"], p["mural_green"], p["mural_blue"],
        p["mural_red"], p["mural_yellow"],
    ]
    random.seed(44)

    for z in range(origin_z - half_l + 1, origin_z + half_l - 1):
        for y in range(5, HEIGHT):
            _, int_w = _arch_profile(y, WIDTH, WALL_T)
            if int_w > 0:
                int_half = int_w // 2
                # Interior surface blocks (just inside the wall)
                for side in [-1, 1]:
                    ix = origin_x + side * (int_half)
                    if world.get_block(ix, y, z) == p["exterior"]:
                        # Replace interior-facing surface with mural
                        # Color depends on position (warm center, green lower, blue upper)
                        if y > 30:
                            color = p["mural_blue"]
                        elif y < 10:
                            color = p["mural_green"]
                        else:
                            color = random.choice(mural_colors)
                        world.set_block(ix, y, z, color)

    # ===== 5. Market floor =====
    market_floor = filled_rectangle(WIDTH - WALL_T * 2 - 2, LENGTH - 4)
    for x, z in market_floor:
        world.set_block(origin_x + x, 0, origin_z + z, p["market_floor"])

    # Market stalls (rows of simple structures)
    for row_z in range(origin_z - half_l + 10, origin_z + half_l - 10, 8):
        for stall_x in range(origin_x - 15, origin_x + 16, 6):
            # Simple stall: counter
            for dx in range(4):
                world.set_block(stall_x + dx, 1, row_z, p["apartment_floor"])

    # ===== 6. Apartment floors inside the arch walls =====
    for y in range(8, 36, 3):
        ext_w, int_w = _arch_profile(y, WIDTH, WALL_T)
        if int_w > 0 and ext_w > int_w:
            wall_depth = (ext_w - int_w) // 2
            # Floor slabs in the wall thickness (apartments)
            for z in range(origin_z - half_l + 1, origin_z + half_l - 1):
                for side in [-1, 1]:
                    ext_half = ext_w // 2
                    int_half = int_w // 2
                    if side == -1:
                        for x in range(origin_x - ext_half + 1, origin_x - int_half):
                            world.set_block(x, y, z, p["apartment_floor"])
                    else:
                        for x in range(origin_x + int_half, origin_x + ext_half - 1):
                            world.set_block(x, y, z, p["apartment_floor"])
