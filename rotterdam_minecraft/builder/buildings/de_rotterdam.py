"""De Rotterdam builder — 3 shifted towers on shared base.

150 blocks tall, 107x36 footprint. Key feature: towers shift at different heights.
"""

from ..engine.world import World
from ..engine.palette import DE_ROTTERDAM, Palette
from ..core.extrusion import extrude_box
from ..patterns.curtain_wall import curtain_wall
from ..patterns.floor_stack import floor_stack
from ..core.shapes import filled_rectangle


def build(world: World, palette: Palette = None, origin_x: int = 0, origin_z: int = 0) -> None:
    """Build De Rotterdam at 1:1 scale (150 blocks)."""
    p = palette or DE_ROTTERDAM

    TOTAL_HEIGHT = 150
    TOWER_WIDTH = 31   # Each tower width (X)
    TOWER_DEPTH = 36   # Each tower depth (Z)
    GAP = 7            # Between towers
    PLINTH_HEIGHT = 30
    FLOOR_HEIGHT = 3   # Alternating 3 and 4

    # Total footprint: 3 towers + 2 gaps = 31*3 + 7*2 = 107
    total_width = TOWER_WIDTH * 3 + GAP * 2

    # Tower X positions (left edge of each tower)
    tower_starts = [
        origin_x - total_width // 2,
        origin_x - total_width // 2 + TOWER_WIDTH + GAP,
        origin_x - total_width // 2 + 2 * (TOWER_WIDTH + GAP),
    ]

    # Tower shift definitions: (y_start, y_end, dz_offset)
    # Each tower shifts at different heights for the "stacked boxes" effect
    tower_shifts = [
        # West tower
        [
            (0, 30, 0),           # Plinth: aligned
            (30, 70, -4),         # Lower block: shift north
            (70, 110, 0),         # Middle block: aligned
            (110, 150, 5),        # Upper block: shift south
        ],
        # Mid tower
        [
            (0, 30, 0),           # Plinth: aligned
            (30, 65, 4),          # Lower block: shift south
            (65, 105, -5),        # Middle block: shift north (most dramatic)
            (105, 150, 2),        # Upper block: slight south
        ],
        # East tower
        [
            (0, 30, 0),           # Plinth: aligned
            (30, 75, 5),          # Lower block: shift south
            (75, 115, -4),        # Middle block: shift north
            (115, 150, 0),        # Upper block: aligned
        ],
    ]

    # ===== 1. Shared plinth (Y 0-29) =====
    plinth = extrude_box(
        total_width, PLINTH_HEIGHT, TOWER_DEPTH,
        start_x=origin_x - total_width // 2, start_y=0,
        start_z=origin_z - TOWER_DEPTH // 2,
        hollow=True, wall_thickness=2,
    )
    world.set_blocks(plinth, p["plinth"])

    # Ground floor entrance hall (taller, Y 0-8)
    for x in range(origin_x - total_width // 2, origin_x + total_width // 2):
        for z in [origin_z - TOWER_DEPTH // 2, origin_z + TOWER_DEPTH // 2]:
            for y in range(1, 9):
                world.set_block(x, y, z, p["plinth_glass"])

    # Plinth floors
    plinth_fp = filled_rectangle(total_width - 4, TOWER_DEPTH - 4)
    plinth_floors = floor_stack(plinth_fp, floor_interval=5, num_floors=6,
                                start_y=0, floor_block=p["floor"])
    world.set_block_dict(plinth_floors)

    # ===== 2. Three towers with shifts (Y 30-149) =====
    zone_palettes = [
        ("office", p["spandrel_office"]),
        ("hotel", p["spandrel_hotel"]),
        ("residential", p["spandrel_residential"]),
    ]

    for tower_idx, (tx_start, shifts) in enumerate(zip(tower_starts, tower_shifts)):
        for shift_block_idx, (y_start, y_end, dz_offset) in enumerate(shifts):
            if y_start < PLINTH_HEIGHT:
                continue  # Plinth already built

            block_height = y_end - y_start
            tz_start = origin_z - TOWER_DEPTH // 2 + dz_offset

            # Determine facade zone for this block
            zone_idx = min(shift_block_idx - 1, len(zone_palettes) - 1)
            zone_idx = max(0, zone_idx)
            _, spandrel = zone_palettes[zone_idx]

            # Build tower block shell
            shell = extrude_box(
                TOWER_WIDTH, block_height, TOWER_DEPTH,
                start_x=tx_start, start_y=y_start, start_z=tz_start,
                hollow=True, wall_thickness=1,
            )
            world.set_blocks(shell, p["glass_facade"])

            # Apply curtain wall pattern on south and north faces
            for face_z, face_dir in [(tz_start, "z"), (tz_start + TOWER_DEPTH - 1, "z")]:
                cw = curtain_wall(
                    width=TOWER_WIDTH, height=block_height,
                    mullion_spacing=2, spandrel_spacing=FLOOR_HEIGHT,
                    start_x=tx_start, start_y=y_start, start_z=face_z,
                    face="z",
                    glass_block=p["glass_facade"],
                    mullion_block=p["mullions"],
                    spandrel_block=spandrel,
                )
                world.set_block_dict(cw)

            # East and west faces
            for face_x in [tx_start, tx_start + TOWER_WIDTH - 1]:
                cw = curtain_wall(
                    width=TOWER_DEPTH, height=block_height,
                    mullion_spacing=2, spandrel_spacing=FLOOR_HEIGHT,
                    start_x=face_x, start_y=y_start, start_z=tz_start,
                    face="x",
                    glass_block=p["glass_facade"],
                    mullion_block=p["mullions"],
                    spandrel_block=spandrel,
                )
                world.set_block_dict(cw)

            # Floors inside this tower block
            floor_fp = filled_rectangle(TOWER_WIDTH - 2, TOWER_DEPTH - 2)
            floors = floor_stack(
                floor_fp, floor_interval=FLOOR_HEIGHT,
                num_floors=block_height // FLOOR_HEIGHT,
                start_y=y_start, floor_block=p["floor"],
                center_x=tx_start + TOWER_WIDTH // 2,
                center_z=tz_start + TOWER_DEPTH // 2,
            )
            world.set_block_dict(floors)

    # ===== 3. Flat roofs =====
    for tower_idx, (tx_start, shifts) in enumerate(zip(tower_starts, tower_shifts)):
        last_shift = shifts[-1]
        dz_offset = last_shift[2]
        tz_start = origin_z - TOWER_DEPTH // 2 + dz_offset
        for x in range(tx_start, tx_start + TOWER_WIDTH):
            for z in range(tz_start, tz_start + TOWER_DEPTH):
                world.set_block(x, TOTAL_HEIGHT, z, p["floor"])
