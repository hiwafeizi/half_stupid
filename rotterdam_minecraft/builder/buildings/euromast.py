"""Euromast builder — tallest structure, simplest geometry.

185 blocks tall. Composed of:
- Base pavilion (diameter 14, 4 blocks)
- Concrete shaft (diameter 10, hollow, 97 blocks)
- Crow's nest disc (diameter 29, asymmetric, 6 blocks thick)
- Space Tower tube (diameter 3, 80 blocks)
- Euroscoop cabin (ring at height ~158)
- Antenna tip (end rods, 5 blocks)
"""

from ..engine.world import World
from ..engine.palette import EUROMAST, Palette
from ..core.shapes import filled_circle, circle_ring, circle_outline
from ..core.extrusion import extrude_cylinder, extrude_ring
from ..core.transforms import translate


def build(world: World, palette: Palette = None, origin_x: int = 0, origin_z: int = 0) -> None:
    """Build the Euromast at 1:1 scale (185 blocks)."""
    p = palette or EUROMAST

    # --- 1. Base pavilion (Y 0-4, diameter 14) ---
    base = extrude_cylinder(radius=7, height=5, start_y=0, hollow=True, wall_thickness=2)
    world.set_blocks(translate(base, origin_x, 0, origin_z), p["base"])

    # Base entrance (cut opening on south side)
    for y in range(0, 4):
        for x in range(-2, 3):
            world.remove_block(origin_x + x, y, origin_z + 7)

    # --- 2. Concrete shaft (Y 5-101, diameter 10, hollow) ---
    shaft = extrude_cylinder(radius=5, height=97, start_y=5, hollow=True, wall_thickness=1)
    world.set_blocks(translate(shaft, origin_x, 0, origin_z), p["shaft"])

    # Interior elevator shaft (2x2 at center)
    for y in range(5, 102):
        world.set_block(origin_x, y, origin_z, p["shaft_interior"])
        world.set_block(origin_x + 1, y, origin_z, p["shaft_interior"])
        world.set_block(origin_x, y, origin_z + 1, p["shaft_interior"])
        world.set_block(origin_x + 1, y, origin_z + 1, p["shaft_interior"])

    # --- 3. Ship's bridge level (Y 28-32) ---
    # Small observation ring around shaft
    bridge_ring = circle_ring(7, 5)
    for x, z in bridge_ring:
        world.set_block(origin_x + x, 30, origin_z + z, p["crows_nest_windows"])
    # Floor
    bridge_floor = filled_circle(7)
    for x, z in bridge_floor:
        world.set_block(origin_x + x, 28, origin_z + z, p["crows_nest_top"])

    # --- 4. Crow's nest (Y 97-105, diameter 29, asymmetric) ---
    # The disc is offset: shaft is in the NE quadrant, larger lobe extends SW
    disc_offset_x = -5  # Shift disc southwest relative to shaft
    disc_offset_z = -5

    # Bottom of disc (underside)
    disc_bottom = filled_circle(14.5)
    for x, z in disc_bottom:
        world.set_block(
            origin_x + x + disc_offset_x, 97,
            origin_z + z + disc_offset_z,
            p["crows_nest_underside"],
        )

    # Disc walls and floors (6 blocks thick: Y 97-102)
    for y in range(97, 103):
        ring = circle_ring(14.5, 13.5)
        for x, z in ring:
            world.set_block(
                origin_x + x + disc_offset_x, y,
                origin_z + z + disc_offset_z,
                p["crows_nest_top"],
            )
        # Windows on the outer ring
        windows = circle_outline(14.5, thickness=1)
        for x, z in windows:
            if y > 98 and y < 102:  # Window band
                world.set_block(
                    origin_x + x + disc_offset_x, y,
                    origin_z + z + disc_offset_z,
                    p["crows_nest_windows"],
                )

    # Disc top (roof)
    for x, z in disc_bottom:
        world.set_block(
            origin_x + x + disc_offset_x, 103,
            origin_z + z + disc_offset_z,
            p["crows_nest_top"],
        )

    # Restaurant floor
    interior = filled_circle(13.5)
    for x, z in interior:
        world.set_block(
            origin_x + x + disc_offset_x, 99,
            origin_z + z + disc_offset_z,
            p["crows_nest_top"],
        )

    # --- 5. Space Tower (Y 105-180, diameter 3) ---
    tower = extrude_cylinder(radius=1.5, height=75, start_y=105)
    world.set_blocks(translate(tower, origin_x, 0, origin_z), p["space_tower"])

    # --- 6. Euroscoop cabin (ring at Y 158-161) ---
    euroscoop_ring = circle_ring(4.5, 2)
    for y in range(158, 162):
        for x, z in euroscoop_ring:
            world.set_block(origin_x + x, y, origin_z + z, p["euroscoop"])

    # --- 7. Antenna tip (Y 180-185) ---
    for y in range(180, 186):
        world.set_block(origin_x, y, origin_z, p["antenna"])
