"""Depot Boijmans Van Beuningen builder — mirrored bowl shape.

40 blocks tall, base diameter 40, top diameter 60.
Uses parabolic curve formula: d(h) = 40 + 20*(h/40)^1.6
"""

from ..engine.world import World
from ..engine.palette import DEPOT_BOIJMANS, Palette
from ..core.shapes import filled_circle, circle_ring, filled_rectangle
from ..core.curves import parabolic_diameter
from ..patterns.tapering import bowl_profile
from ..patterns.sloped_roof import flat_roof
import random


def build(world: World, palette: Palette = None, origin_x: int = 0, origin_z: int = 0) -> None:
    """Build Depot Boijmans at 1:1 scale (40 blocks, 40→60 diameter)."""
    p = palette or DEPOT_BOIJMANS

    HEIGHT = 40
    BASE_D = 40.0
    TOP_D = 60.0
    EXPONENT = 1.6
    WALL_THICKNESS = 2
    FLOOR_INTERVAL = 6
    ATRIUM_W = 6
    ATRIUM_D = 18

    # ===== 1. Main bowl structure =====
    bowl = bowl_profile(
        base_diameter=BASE_D, top_diameter=TOP_D,
        height=HEIGHT, exponent=EXPONENT,
        wall_thickness=WALL_THICKNESS,
        start_y=0,
        wall_block=p["facade_primary"],
        floor_block=p["floor"],
        floor_interval=FLOOR_INTERVAL,
        atrium_width=ATRIUM_W, atrium_depth=ATRIUM_D,
    )
    # Translate to origin
    translated = {}
    for (x, y, z), block in bowl.items():
        translated[(x + origin_x, y, z + origin_z)] = block
    world.set_block_dict(translated)

    # ===== 2. Mirrored facade variation =====
    # Add accent blocks to break up the mirror surface
    diameters = parabolic_diameter(BASE_D, TOP_D, HEIGHT, EXPONENT)
    accent_blocks = [p["facade_accent_1"], p["facade_accent_2"], p["facade_accent_3"]]
    random.seed(42)  # Reproducible pattern

    for y in range(HEIGHT):
        d = diameters[y]
        r = d / 2
        outline = set()
        # Get outer ring points
        for x in range(int(-r) - 1, int(r) + 2):
            for z in range(int(-r) - 1, int(r) + 2):
                dist_sq = x * x + z * z
                if (r - 1) ** 2 <= dist_sq <= r ** 2:
                    outline.add((x, z))

        for x, z in outline:
            if random.random() < 0.15:  # 15% accent variation
                accent = random.choice(accent_blocks)
                world.set_block(origin_x + x, y, origin_z + z, accent)

    # ===== 3. Central atrium with criss-crossing staircases =====
    # Clear atrium void
    for y in range(1, HEIGHT):
        for x in range(-ATRIUM_W // 2, ATRIUM_W // 2 + 1):
            for z in range(-ATRIUM_D // 2, ATRIUM_D // 2 + 1):
                world.remove_block(origin_x + x, y, origin_z + z)

    # Zigzag staircases (5 flights alternating direction)
    stair_y = 1
    direction = 1  # 1 = south, -1 = north
    while stair_y < HEIGHT - 2:
        stair_z_start = -ATRIUM_D // 2 if direction == 1 else ATRIUM_D // 2
        stair_z_end = ATRIUM_D // 2 if direction == 1 else -ATRIUM_D // 2

        steps = abs(stair_z_end - stair_z_start)
        rise = min(FLOOR_INTERVAL, HEIGHT - stair_y)

        for step in range(min(steps, rise * 2)):
            sz = stair_z_start + step * direction
            sy = stair_y + step // 2
            if sy < HEIGHT:
                world.set_block(origin_x - 1, sy, origin_z + sz, p["atrium_stairs"])
                world.set_block(origin_x, sy, origin_z + sz, p["atrium_stairs"])
                world.set_block(origin_x + 1, sy, origin_z + sz, p["atrium_stairs"])

        stair_y += rise
        direction *= -1

    # Glass elevator shafts (2, on sides of atrium)
    for ez in [-ATRIUM_D // 2, ATRIUM_D // 2]:
        for y in range(0, HEIGHT):
            world.set_block(origin_x - 2, y, origin_z + ez, p["atrium_glass"])
            world.set_block(origin_x + 2, y, origin_z + ez, p["atrium_glass"])

    # ===== 4. Entrance (south side, ground level) =====
    entrance_y_top = 5
    for x in range(-4, 5):
        for y in range(0, entrance_y_top):
            world.set_block(origin_x + x, y, origin_z + int(BASE_D / 2), p["atrium_glass"])

    # ===== 5. Rooftop garden (Y = 35-40) =====
    top_r = TOP_D / 2
    rooftop = filled_circle(top_r - 1)  # Slightly inset from edge

    # Grass surface
    for x, z in rooftop:
        world.set_block(origin_x + x, HEIGHT, origin_z + z, p["rooftop_grass"])

    # Paths (cross pattern)
    for x in range(-2, 3):
        for z in range(int(-top_r) + 2, int(top_r) - 1):
            world.set_block(origin_x + x, HEIGHT, origin_z + z, p["rooftop_path"])
    for z in range(-2, 3):
        for x in range(int(-top_r) + 2, int(top_r) - 1):
            world.set_block(origin_x + x, HEIGHT, origin_z + z, p["rooftop_path"])

    # Trees (birch + spruce scattered)
    random.seed(43)
    tree_positions = []
    for x, z in rooftop:
        if abs(x) > 3 and abs(z) > 3:  # Not on paths
            dist_sq = x * x + z * z
            if dist_sq < (top_r - 3) ** 2:
                tree_positions.append((x, z))

    random.shuffle(tree_positions)
    # Place 75 birch trees
    for i, (x, z) in enumerate(tree_positions[:75]):
        trunk_h = random.randint(4, 6)
        for ty in range(1, trunk_h + 1):
            world.set_block(origin_x + x, HEIGHT + ty, origin_z + z, "minecraft:birch_log")
        # Simple leaf canopy
        for lx in range(-2, 3):
            for lz in range(-2, 3):
                for ly in range(trunk_h - 1, trunk_h + 2):
                    if abs(lx) + abs(lz) <= 3:
                        world.set_block(
                            origin_x + x + lx, HEIGHT + ly,
                            origin_z + z + lz, "minecraft:birch_leaves"
                        )

    # Glass railing around edge
    edge = set()
    for x in range(int(-top_r) - 1, int(top_r) + 2):
        for z in range(int(-top_r) - 1, int(top_r) + 2):
            dist_sq = x * x + z * z
            if (top_r - 1) ** 2 <= dist_sq <= top_r ** 2:
                edge.add((x, z))
    for x, z in edge:
        world.set_block(origin_x + x, HEIGHT + 1, origin_z + z, p["railing"])
