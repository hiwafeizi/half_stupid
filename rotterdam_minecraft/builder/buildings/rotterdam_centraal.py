"""Rotterdam Centraal builder — asymmetric pointed canopy + platforms.

30-block peak, 250x150 platform area, 12 tracks.
"""

from ..engine.world import World
from ..engine.palette import ROTTERDAM_CENTRAAL, Palette
from ..core.shapes import filled_rectangle
from ..patterns.curtain_wall import curtain_wall
from ..patterns.sloped_roof import asymmetric_peak, flat_roof
from ..patterns.facade_detail import clock_face
from ..patterns.floor_stack import floor_stack


def build(world: World, palette: Palette = None, origin_x: int = 0, origin_z: int = 0) -> None:
    """Build Rotterdam Centraal at 1:1 scale."""
    p = palette or ROTTERDAM_CENTRAAL

    FACADE_WIDTH = 55
    PEAK_HEIGHT = 30
    PEAK_X = 18  # 1/3 from left
    HALL_DEPTH = 35
    PLATFORM_LENGTH = 250
    PLATFORM_WIDTH = 150
    PLATFORM_ROOF_H = 12

    half_fw = FACADE_WIDTH // 2

    # ===== 1. Asymmetric pointed canopy (front facade) =====
    canopy = asymmetric_peak(
        width=FACADE_WIDTH, depth=HALL_DEPTH,
        peak_height=PEAK_HEIGHT, peak_x_position=PEAK_X,
        start_x=origin_x - half_fw, start_y=0, start_z=origin_z,
        block=p["canopy"],
    )
    world.set_block_dict(canopy)

    # Canopy overhang (extends 8 blocks forward)
    for x in range(origin_x - half_fw, origin_x + half_fw - FACADE_WIDTH + FACADE_WIDTH):
        # Calculate height at this x
        if x - (origin_x - half_fw) <= PEAK_X:
            h = round(PEAK_HEIGHT * (x - (origin_x - half_fw)) / PEAK_X) if PEAK_X > 0 else PEAK_HEIGHT
        else:
            remaining = FACADE_WIDTH - PEAK_X - 1
            h = round(PEAK_HEIGHT * (FACADE_WIDTH - 1 - (x - (origin_x - half_fw))) / remaining) if remaining > 0 else 0
        if h > 3:
            for dz in range(-8, 0):
                world.set_block(x, h, origin_z + dz, p["canopy"])

    # ===== 2. Glass curtain wall (behind canopy) =====
    glass_wall = curtain_wall(
        width=FACADE_WIDTH, height=min(PEAK_HEIGHT - 2, 25),
        mullion_spacing=3, spandrel_spacing=5,
        start_x=origin_x - half_fw, start_y=0, start_z=origin_z + 1,
        face="z",
        glass_block=p["glass_facade"], mullion_block=p["columns"],
        spandrel_block=p["floor"],
    )
    world.set_block_dict(glass_wall)

    # ===== 3. Clock =====
    clock = clock_face(
        diameter=8,
        start_x=origin_x, start_y=18, start_z=origin_z + 1,
        face="z",
        face_block=p["clock_face"], rim_block=p["clock_rim"],
    )
    world.set_block_dict(clock)

    # ===== 4. Entrance hall interior =====
    # Wooden slat ceiling
    for x in range(origin_x - half_fw + 2, origin_x + half_fw - FACADE_WIDTH + FACADE_WIDTH - 2):
        for z in range(origin_z + 2, origin_z + HALL_DEPTH - 2):
            # Ceiling follows the canopy slope (interior)
            rel_x = x - (origin_x - half_fw)
            if rel_x <= PEAK_X:
                h = round(PEAK_HEIGHT * rel_x / max(1, PEAK_X)) - 1
            else:
                remaining = FACADE_WIDTH - PEAK_X - 1
                h = round(PEAK_HEIGHT * (FACADE_WIDTH - 1 - rel_x) / max(1, remaining)) - 1
            if h > 5:
                # Wooden slats (alternating wood types)
                if z % 3 == 0:
                    world.set_block(x, h, z, p["wooden_ceiling"])
                elif z % 3 == 1:
                    world.set_block(x, h, z, p["wooden_accent"])
                else:
                    world.set_block(x, h, z, p["wooden_ceiling"])

    # Floor
    for x in range(origin_x - half_fw, origin_x + half_fw - FACADE_WIDTH + FACADE_WIDTH):
        for z in range(origin_z, origin_z + HALL_DEPTH):
            world.set_block(x, 0, z, p["floor"])

    # ===== 5. Platform area (behind entrance hall) =====
    platform_start_z = origin_z + HALL_DEPTH

    # Platform floor
    for x in range(origin_x - PLATFORM_WIDTH // 2, origin_x + PLATFORM_WIDTH // 2):
        for z in range(platform_start_z, platform_start_z + PLATFORM_LENGTH):
            world.set_block(x, 0, z, p["platform"])

    # Track layout: 12 tracks with platforms between
    track_positions = []
    current_x = origin_x - 57  # Start of track area
    for i in range(12):
        # Track (5 blocks wide)
        for z in range(platform_start_z, platform_start_z + PLATFORM_LENGTH):
            for dx in range(5):
                world.set_block(current_x + dx, 0, z, p["tracks"])
        track_positions.append(current_x)
        current_x += 5

        # Platform between tracks (6 blocks wide), except after last track
        if i < 11:
            for z in range(platform_start_z, platform_start_z + PLATFORM_LENGTH):
                for dx in range(4):
                    world.set_block(current_x + dx, 0, z, p["platform"])
                    world.set_block(current_x + dx, 1, z, p["platform"])
            current_x += 4

    # Platform roof (flat glass + solar panels)
    for x in range(origin_x - PLATFORM_WIDTH // 2, origin_x + PLATFORM_WIDTH // 2):
        for z in range(platform_start_z, platform_start_z + PLATFORM_LENGTH):
            if (x + z) % 3 == 0:
                world.set_block(x, PLATFORM_ROOF_H, z, p["solar_panels"])
            else:
                world.set_block(x, PLATFORM_ROOF_H, z, p["roof_glass"])

    # Roof support columns (Y-shaped, every 20 blocks)
    for x in range(origin_x - PLATFORM_WIDTH // 2 + 10, origin_x + PLATFORM_WIDTH // 2, 25):
        for z in range(platform_start_z + 10, platform_start_z + PLATFORM_LENGTH, 20):
            # Column shaft
            for y in range(1, PLATFORM_ROOF_H):
                world.set_block(x, y, z, p["columns"])
