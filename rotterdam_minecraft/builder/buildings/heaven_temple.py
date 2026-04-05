"""Heaven Temple — raised temple, water moat with fish, + glass, X lava, pyramid.

Design:
    - Temple on 5-block quartz platform (hollow)
    - Water moat flush with platform top (water surface = platform surface)
    - Stairs extend BEYOND the water, stepping down outward from + paths
    - Stairs face correct direction (toward the building)
    - + shape: glass corridors, open entries
    - X shape: lava sealed in glass
    - Center: red temple, lava core, pyramid roof
"""

from __future__ import annotations
from ..engine.world import World


RED = "minecraft:redstone_block"
GLASS = "minecraft:glass"
LAVA = "minecraft:lava"
WHITE = "minecraft:quartz_block"
WATER = "minecraft:water"
SEA_LANTERN = "minecraft:sea_lantern"
GLOWSTONE = "minecraft:glowstone"

# Directional stairs (face the direction you walk UP toward)
STAIRS_N = "minecraft:quartz_stairs_south"  # player walks south (toward building) going up
STAIRS_S = "minecraft:quartz_stairs_north"  # player walks north going up
STAIRS_W = "minecraft:quartz_stairs_east"   # player walks east going up
STAIRS_E = "minecraft:quartz_stairs_west"   # player walks west going up


def _classify(x, z, half, arm_half):
    if abs(x) > half or abs(z) > half:
        return "outside"
    if abs(x) <= arm_half and abs(z) <= half:
        return "plus"
    if abs(z) <= arm_half and abs(x) <= half:
        return "plus"
    dd = abs(abs(x) - abs(z))
    dr = min(abs(x), abs(z))
    if dr > arm_half:
        if dd <= 2:
            return "x_lava"
        elif dd <= 5:
            return "x_glass"
    return "fill"


def build(world: World, palette=None, origin_x: int = 0, origin_z: int = 0) -> None:
    half = 75
    arm_half = 10
    center_half = 22
    wall_h = 28
    plat_h = 5
    water_border = 12
    stair_len = 5       # stairs extend 5 blocks beyond water

    ox, oz = origin_x, origin_z
    total_half = half + water_border
    stair_half = total_half + stair_len
    by = plat_h

    # =========================================
    # PASS 0: Platform — hollow shell, white quartz
    # =========================================
    # Top surface
    for x in range(-half, half + 1):
        for z in range(-half, half + 1):
            world.set_block(ox + x, by, oz + z, WHITE)
    # Bottom surface
    for x in range(-half, half + 1):
        for z in range(-half, half + 1):
            world.set_block(ox + x, 0, oz + z, WHITE)
    # 4 outer walls of platform
    for y in range(0, by + 1):
        for x in range(-half, half + 1):
            world.set_block(ox + x, y, oz - half, WHITE)
            world.set_block(ox + x, y, oz + half, WHITE)
        for z in range(-half, half + 1):
            world.set_block(ox - half, y, oz + z, WHITE)
            world.set_block(ox + half, y, oz + z, WHITE)

    # =========================================
    # PASS 0.5: Water moat — water surface at y=plat_h (flush with platform)
    # Only in 4 quadrant areas, not on + paths
    # Quartz walls on path edges and outer edge
    # =========================================
    for x in range(-total_half, total_half + 1):
        for z in range(-total_half, total_half + 1):
            in_temple = abs(x) <= half and abs(z) <= half
            if in_temple:
                continue
            in_plus = abs(x) <= arm_half or abs(z) <= arm_half

            if in_plus:
                # Dry + path bridge — solid white at platform height
                for y in range(0, by + 1):
                    world.set_block(ox + x, y, oz + z, WHITE)
                continue

            # Containment walls: outer edge, path edges, temple edge
            is_outer = abs(x) == total_half or abs(z) == total_half
            is_path_wall = abs(x) == arm_half + 1 or abs(z) == arm_half + 1

            if is_outer or is_path_wall:
                for y in range(0, by + 1):
                    world.set_block(ox + x, y, oz + z, WHITE)
            else:
                # Pool: quartz floor, water up to platform level
                world.set_block(ox + x, 0, oz + z, WHITE)
                for wy in range(1, by + 1):
                    world.set_block(ox + x, wy, oz + z, WATER)

    # =========================================
    # PASS 1: Stairs — BEYOND the water, extending outward
    # Bridge is at y=plat_h (y=5). Stairs step down from y=4 to y=0.
    # Bottom stair sits on ground (y=0), top stair (y=4) meets bridge (y=5).
    # Player walks up stairs onto the bridge seamlessly.
    # =========================================
    # plat_h+1 steps so top step is flush with platform (y=5) and bottom is ground (y=0)
    num_steps = plat_h + 1  # 6 steps: y=5, y=4, y=3, y=2, y=1, y=0

    for step in range(num_steps):
        # step 0 = closest to building (flush with platform), step 5 = ground
        stair_y = plat_h - step           # y=5, 4, 3, 2, 1, 0
        dist = total_half + 1 + step      # each step 1 block further out

        # North: at -z, player walks +z (south) to go up
        for x in range(-arm_half, arm_half + 1):
            world.set_block(ox + x, stair_y, oz - dist, STAIRS_N)
            # Solid support below the stair
            for fy in range(0, stair_y):
                world.set_block(ox + x, fy, oz - dist, WHITE)

        # South: at +z, player walks -z (north) to go up
        for x in range(-arm_half, arm_half + 1):
            world.set_block(ox + x, stair_y, oz + dist, STAIRS_S)
            for fy in range(0, stair_y):
                world.set_block(ox + x, fy, oz + dist, WHITE)

        # West: at -x, player walks +x (east) to go up
        for z in range(-arm_half, arm_half + 1):
            world.set_block(ox - dist, stair_y, oz + z, STAIRS_W)
            for fy in range(0, stair_y):
                world.set_block(ox - dist, fy, oz + z, WHITE)

        # East: at +x, player walks -x (west) to go up
        for z in range(-arm_half, arm_half + 1):
            world.set_block(ox + dist, stair_y, oz + z, STAIRS_E)
            for fy in range(0, stair_y):
                world.set_block(ox + dist, fy, oz + z, WHITE)

    # =========================================
    # PASS 2: Fill zones — red rooms (shell only)
    # =========================================
    for x in range(-half, half + 1):
        for z in range(-half, half + 1):
            kind = _classify(x, z, half, arm_half)
            if kind != "fill":
                continue
            if abs(x) <= center_half and abs(z) <= center_half:
                continue

            dist = max(abs(x), abs(z))
            local_h = max(8, int(wall_h * (1.2 - dist / half)))

            is_outer = abs(x) == half or abs(z) == half
            is_edge = False
            for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if _classify(x + dx, z + dz, half, arm_half) != "fill":
                    is_edge = True
                    break

            if is_outer or is_edge:
                for y in range(by + 1, by + local_h + 1):
                    world.set_block(ox + x, y, oz + z, RED)
            else:
                on_wall = (abs(x) % 12 == 0) or (abs(z) % 12 == 0)
                if on_wall:
                    for y in range(by + 1, by + local_h + 1):
                        world.set_block(ox + x, y, oz + z, RED)
                else:
                    world.set_block(ox + x, by + local_h, oz + z, RED)
                    if abs(x) % 12 == 6 and abs(z) % 12 == 6:
                        world.set_block(ox + x, by + local_h - 1, oz + z, GLOWSTONE)

    # =========================================
    # PASS 3: X lava channels sealed in glass
    # =========================================
    for x in range(-half, half + 1):
        for z in range(-half, half + 1):
            kind = _classify(x, z, half, arm_half)
            dr = min(abs(x), abs(z))
            if dr <= center_half:
                continue

            at_edge = abs(x) >= half - 2 or abs(z) >= half - 2

            if kind == "x_lava":
                world.set_block(ox + x, by, oz + z, GLASS)
                if at_edge:
                    for y in range(by + 1, by + wall_h):
                        world.set_block(ox + x, y, oz + z, GLASS)
                else:
                    for y in range(by + 1, by + wall_h - 1):
                        world.set_block(ox + x, y, oz + z, LAVA)
                    world.set_block(ox + x, by + wall_h - 1, oz + z, GLASS)
            elif kind == "x_glass":
                for y in range(by + 1, by + wall_h):
                    world.set_block(ox + x, y, oz + z, GLASS)

    # =========================================
    # PASS 4: + glass corridor arms
    # =========================================
    for x in range(-half, half + 1):
        for z in range(-half, half + 1):
            if _classify(x, z, half, arm_half) != "plus":
                continue
            if abs(x) <= center_half and abs(z) <= center_half:
                continue

            in_ns = abs(x) <= arm_half and abs(z) > arm_half
            in_ew = abs(z) <= arm_half and abs(x) > arm_half

            if in_ns:
                is_wall = abs(x) == arm_half
            elif in_ew:
                is_wall = abs(z) == arm_half
            else:
                continue

            if is_wall:
                for y in range(by + 1, by + wall_h + 1):
                    world.set_block(ox + x, y, oz + z, GLASS)
                dist = max(abs(x), abs(z))
                if dist % 15 == 0:
                    for y in range(by + 1, by + wall_h + 1):
                        world.set_block(ox + x, y, oz + z, RED)

            world.set_block(ox + x, by + wall_h, oz + z, GLASS)

            if in_ns and x == 0 and abs(z) % 12 == 0:
                world.set_block(ox + x, by + wall_h, oz + z, SEA_LANTERN)
            if in_ew and z == 0 and abs(x) % 12 == 0:
                world.set_block(ox + x, by + wall_h, oz + z, SEA_LANTERN)

    # =========================================
    # PASS 5: Central temple — red walls, lava in glass, + walkway, pyramid
    # =========================================
    ch = center_half
    corridor_half = arm_half

    for x in range(-ch, ch + 1):
        for z in range(-ch, ch + 1):
            is_wall = abs(x) == ch or abs(z) == ch
            is_corner = abs(x) == ch and abs(z) == ch
            in_corridor = abs(x) <= corridor_half or abs(z) <= corridor_half

            if is_wall:
                for y in range(by + 1, by + wall_h + 1):
                    world.set_block(ox + x, y, oz + z, RED)
                if abs(x) <= arm_half and (z == ch or z == -ch):
                    for y in range(by + 1, by + wall_h - 3):
                        world.set_block(ox + x, y, oz + z, "minecraft:air")
                if abs(z) <= arm_half and (x == ch or x == -ch):
                    for y in range(by + 1, by + wall_h - 3):
                        world.set_block(ox + x, y, oz + z, "minecraft:air")

            elif in_corridor:
                is_corridor_edge = False
                if abs(x) <= corridor_half and abs(z) > corridor_half:
                    is_corridor_edge = abs(x) == corridor_half
                elif abs(z) <= corridor_half and abs(x) > corridor_half:
                    is_corridor_edge = abs(z) == corridor_half
                if is_corridor_edge and not (abs(x) <= corridor_half and abs(z) <= corridor_half):
                    for y in range(by + 1, by + wall_h):
                        world.set_block(ox + x, y, oz + z, GLASS)
                world.set_block(ox + x, by + wall_h, oz + z, GLASS)
                if abs(x) % 8 == 0 and abs(z) == 0:
                    world.set_block(ox + x, by + wall_h, oz + z, SEA_LANTERN)
                if abs(z) % 8 == 0 and abs(x) == 0:
                    world.set_block(ox + x, by + wall_h, oz + z, SEA_LANTERN)

            else:
                # Lava quadrants
                world.set_block(ox + x, by, oz + z, GLASS)
                for y in range(by + 1, by + wall_h - 1):
                    world.set_block(ox + x, y, oz + z, LAVA)
                world.set_block(ox + x, by + wall_h - 1, oz + z, GLASS)

            if is_corner:
                for y in range(by + wall_h + 1, by + wall_h + 6):
                    world.set_block(ox + x, y, oz + z, RED)

    # Pyramid roof
    py = by + wall_h + 1
    layer = 0
    while ch - layer > 0:
        s = ch - layer
        for x in range(-s, s + 1):
            for z in range(-s, s + 1):
                world.set_block(ox + x, py + layer, oz + z, RED)
        layer += 1

    world.set_block(ox, py + layer, oz, GLASS)
    world.set_block(ox, py + layer + 1, oz, SEA_LANTERN)
