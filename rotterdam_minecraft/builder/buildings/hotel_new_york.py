"""Hotel New York builder — Art Nouveau waterfront building with twin towers.

85x28 footprint, 17 blocks main body, 38-block twin towers.
"""

from ..engine.world import World
from ..engine.palette import HOTEL_NEW_YORK, Palette
from ..core.shapes import filled_rectangle, rectangle_shell, octagon
from ..core.extrusion import extrude_box
from ..patterns.window_grid import window_grid
from ..patterns.sloped_roof import hipped_roof, conical_cap
from ..patterns.facade_detail import balcony, cornice_band
from ..patterns.floor_stack import floor_stack


def build(world: World, palette: Palette = None, origin_x: int = 0, origin_z: int = 0) -> None:
    """Build Hotel New York at 1:1 scale."""
    p = palette or HOTEL_NEW_YORK

    LENGTH = 85   # X axis
    WIDTH = 28    # Z axis
    MAIN_H = 17
    TOWER_H = 38
    FLOOR_H = 4

    half_l = LENGTH // 2
    half_w = WIDTH // 2

    # ===== 1. Main body (85x28x17) =====
    main_shell = extrude_box(
        LENGTH, MAIN_H, WIDTH,
        start_x=origin_x - half_l, start_y=0, start_z=origin_z - half_w,
        hollow=True, wall_thickness=1,
    )
    world.set_blocks(main_shell, p["walls"])

    # Ground floor: stone base (first 5 blocks)
    for x in range(origin_x - half_l, origin_x + half_l):
        for z in range(origin_z - half_w, origin_z + half_w):
            for y in range(0, 5):
                if world.has_block(x, y, z):
                    world.set_block(x, y, z, p["stone_base"])

    # Windows on all facades
    for face_dir, sx, sz, fw in [
        ("z", origin_x - half_l, origin_z - half_w, LENGTH),
        ("z", origin_x - half_l, origin_z + half_w - 1, LENGTH),
        ("x", origin_x - half_l, origin_z - half_w, WIDTH),
        ("x", origin_x + half_l - 1, origin_z - half_w, WIDTH),
    ]:
        wall = window_grid(
            wall_width=fw, wall_height=MAIN_H,
            window_width=1, window_height=2,
            h_spacing=4, v_spacing=FLOOR_H,
            margin_x=2, margin_y=5,
            start_x=sx, start_y=0, start_z=sz,
            face=face_dir,
            wall_block=p["walls"], window_block=p["windows"],
        )
        world.set_block_dict(wall)

    # Cornice at roofline
    cornice = cornice_band(
        LENGTH, WIDTH,
        start_x=origin_x, start_y=MAIN_H, start_z=origin_z,
        block=p["stone_details"], extend_out=1,
    )
    world.set_block_dict(cornice)

    # Interior floors
    interior_fp = filled_rectangle(LENGTH - 2, WIDTH - 2)
    floors = floor_stack(interior_fp, floor_interval=FLOOR_H, num_floors=4,
                         start_y=0, floor_block=p["floor"])
    world.set_block_dict(floors)

    # ===== 2. Hipped roof on main body =====
    roof = hipped_roof(
        LENGTH, WIDTH, height=6,
        start_x=origin_x, start_y=MAIN_H, start_z=origin_z,
        block=p["main_roof"],
    )
    world.set_block_dict(roof)

    # ===== 3. Twin towers (octagonal, 38 blocks) =====
    tower_positions = [
        (origin_x - half_l + 5, origin_z),   # West tower
        (origin_x + half_l - 6, origin_z),   # East tower
    ]

    for ti, (tx, tz) in enumerate(tower_positions):
        tower_d = 7

        # Tower shaft
        oct_profile = octagon(tower_d)
        for y in range(0, TOWER_H - 8):
            for x, z in oct_profile:
                world.set_block(tx + x, y, tz + z, p["walls"])

            # Tower windows (every floor)
            if y % FLOOR_H >= 2 and y > 5:
                for x, z in oct_profile:
                    # Only on outer ring
                    if abs(x) == tower_d // 2 or abs(z) == tower_d // 2:
                        world.set_block(tx + x, y, tz + z, p["windows"])

        # Copper dome/cupola (top 8 blocks)
        cupola = conical_cap(
            base_diameter=tower_d + 2, height=9,
            start_x=tx, start_y=TOWER_H - 8, start_z=tz,
            block=p["tower_roofs"],
        )
        world.set_block_dict(cupola)

        # Finial
        world.set_block(tx, TOWER_H + 1, tz, p["iron_details"])

        # Clock on north tower (ti == 0)
        if ti == 0:
            from ..patterns.facade_detail import clock_face
            clock = clock_face(
                diameter=4,
                start_x=tx, start_y=22, start_z=tz - tower_d // 2 - 1,
                face="z",
            )
            world.set_block_dict(clock)

    # ===== 4. Balconies (ship-railing style) =====
    # West facade balconies
    for bx in range(origin_x - half_l + 15, origin_x + half_l - 15, 12):
        for floor_y in [9, 13]:
            b = balcony(
                width=4, depth=2,
                start_x=bx, start_y=floor_y, start_z=origin_z - half_w,
                face="z",
                floor_block=p["stone_details"], railing_block=p["balconies"],
            )
            world.set_block_dict(b)

    # ===== 5. Gold lettering area (west facade) =====
    # "HOLLAND AMERIKA LIJN" — represented by gold blocks on the facade
    letter_y = 14
    for x in range(origin_x - 20, origin_x + 21, 2):
        world.set_block(x, letter_y, origin_z - half_w, p["gold_lettering"])

    # ===== 6. Waterfront =====
    # Water on 3 sides (south, east, west)
    for x in range(origin_x - half_l - 10, origin_x + half_l + 10):
        for z in range(origin_z + half_w, origin_z + half_w + 15):
            world.set_block(x, -1, z, "minecraft:water")
    for z in range(origin_z - half_w - 10, origin_z + half_w + 15):
        world.set_block(origin_x - half_l - 5, -1, z, "minecraft:water")
        world.set_block(origin_x + half_l + 5, -1, z, "minecraft:water")
