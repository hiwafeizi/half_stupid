"""Erasmus Bridge builder — cable-stayed bridge with bent pylon.

802 blocks long, 139-block pylon, 34 blocks wide.
Key challenge: pylon kink curve and cable fan system.
"""

from ..engine.world import World
from ..engine.palette import ERASMUS_BRIDGE, Palette
from ..core.shapes import filled_rectangle
from ..core.extrusion import extrude_box, extrude_constant
from ..core.curves import bresenham_line_3d
from ..patterns.cables import fan_cables, converging_cables


def build(world: World, palette: Palette = None, origin_x: int = 0, origin_z: int = 0) -> None:
    """Build the Erasmus Bridge at 1:1 scale (802 blocks long)."""
    p = palette or ERASMUS_BRIDGE

    TOTAL_LENGTH = 802
    DECK_WIDTH = 34
    DECK_Y = 17  # Clearance above water
    PYLON_X = origin_x + 284  # Position along bridge length
    PYLON_HEIGHT = 139

    # Bridge runs along X axis, width along Z
    bridge_start_x = origin_x
    bridge_end_x = origin_x + TOTAL_LENGTH

    # ===== 1. Water surface =====
    # (Optional: place water below the bridge)

    # ===== 2. Deck =====
    half_w = DECK_WIDTH // 2
    # Deck surface (1 block thick)
    for x in range(bridge_start_x, bridge_end_x):
        for z in range(origin_z - half_w, origin_z + half_w):
            # Determine lane type based on Z position
            dz = z - (origin_z - half_w)
            if dz < 3 or dz >= DECK_WIDTH - 3:
                # Pedestrian paths
                world.set_block(x, DECK_Y, z, p["deck_sides"])
            elif dz < 6 or dz >= DECK_WIDTH - 6:
                # Cycle paths
                world.set_block(x, DECK_Y, z, "minecraft:red_concrete")
            elif dz == DECK_WIDTH // 2 or dz == DECK_WIDTH // 2 - 1:
                # Center median (cable attachment line)
                world.set_block(x, DECK_Y, z, p["deck_sides"])
            else:
                # Road surface
                world.set_block(x, DECK_Y, z, p["deck_surface"])

    # Deck underside girders (2 blocks deep)
    for x in range(bridge_start_x, bridge_end_x):
        for z in [origin_z - half_w + 5, origin_z + half_w - 6]:
            world.set_block(x, DECK_Y - 1, z, p["deck_sides"])
            world.set_block(x, DECK_Y - 2, z, p["deck_sides"])

    # Railings on both edges
    for x in range(bridge_start_x, bridge_end_x):
        world.set_block(x, DECK_Y + 1, origin_z - half_w, p["railings"])
        world.set_block(x, DECK_Y + 1, origin_z + half_w - 1, p["railings"])

    # Tram tracks
    for x in range(bridge_start_x, bridge_end_x):
        for z_offset in [-4, -3, 3, 4]:
            world.set_block(x, DECK_Y, origin_z + z_offset, p["tram_tracks"])

    # ===== 3. Support piers (5 piers on approach spans) =====
    pier_spacing = 125
    for pier_num in range(5):
        pier_x = bridge_start_x + 50 + pier_num * pier_spacing
        if abs(pier_x - PYLON_X) < 30:
            continue  # Skip if too close to pylon

        for z in range(origin_z - 5, origin_z + 6):
            for y in range(0, DECK_Y):
                if abs(z - origin_z) < 3:
                    world.set_block(pier_x, y, z, p["piers"])
                    world.set_block(pier_x + 1, y, z, p["piers"])

    # ===== 4. Pylon ("The Swan") =====
    pylon_z = origin_z  # Centered on bridge

    # Base section (horizontal, Y 0-19, runs along X)
    for y in range(0, 20):
        for x in range(PYLON_X - 12, PYLON_X + 13):
            for z in range(pylon_z - 5, pylon_z + 6):
                world.set_block(x, y, z, p["pylon"])

    # Kink section (angled, Y 20-49, the "bend" of the swan neck)
    for step in range(30):
        y = 20 + step
        # Move north (negative X direction) as we go up — angled at ~45°
        x_offset = -step
        for dx in range(-4, 5):
            for dz in range(-4, 5):
                world.set_block(PYLON_X + x_offset + dx, y, pylon_z + dz, p["pylon"])

    # Vertical section (Y 50-130, near-vertical with slight lean)
    kink_end_x = PYLON_X - 30
    for step in range(80):
        y = 50 + step
        # Very slight lean: 1 block per 10 height
        x_offset = -(step // 10)
        for dx in range(-3, 4):
            for dz in range(-3, 4):
                world.set_block(kink_end_x + x_offset + dx, y, pylon_z + dz, p["pylon"])

    # Tip (Y 130-138, tapering)
    tip_x = kink_end_x - 8
    for step in range(9):
        y = 130 + step
        size = max(1, 3 - step // 3)
        for dx in range(-size, size + 1):
            for dz in range(-size, size + 1):
                world.set_block(tip_x + dx, y, pylon_z + dz, p["pylon"])

    # ===== 5. Cable system =====
    # Anchor point at pylon top
    anchor_x = tip_x
    anchor_y = 138
    anchor_z = pylon_z

    # Main span cables (32 cables, fan from pylon to south deck)
    main_cables = fan_cables(
        anchor_x=anchor_x, anchor_y=anchor_y, anchor_z=anchor_z,
        deck_y=DECK_Y + 1, deck_z=origin_z,
        deck_start_x=PYLON_X + 10,
        deck_end_x=PYLON_X + 410,
        count=32, block=p["cables"],
    )
    world.set_block_dict(main_cables)

    # Backstay cables (8 cables to north anchor)
    backstay_targets = [
        (PYLON_X - 10 - i * 10, DECK_Y + 1, origin_z)
        for i in range(8)
    ]
    backstays = converging_cables(
        anchor_x=anchor_x, anchor_y=anchor_y, anchor_z=anchor_z,
        targets=backstay_targets, block=p["cables"],
    )
    world.set_block_dict(backstays)

    # ===== 6. Bascule section (southern end, 89 blocks) =====
    bascule_start = bridge_end_x - 89
    # Gap in the deck (2 blocks wide) to show the split
    for z in range(origin_z - half_w, origin_z + half_w):
        world.remove_block(bascule_start, DECK_Y, z)
        world.remove_block(bascule_start + 1, DECK_Y, z)

    # Bascule pier (heavier pier at the hinge point)
    for y in range(0, DECK_Y):
        for x in range(bascule_start - 3, bascule_start + 4):
            for z in range(origin_z - 6, origin_z + 7):
                if abs(z - origin_z) < 5:
                    world.set_block(x, y, z, p["piers"])
