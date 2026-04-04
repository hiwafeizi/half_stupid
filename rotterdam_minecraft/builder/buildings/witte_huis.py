"""Witte Huis (White House) builder — 3-zone Art Nouveau tower.

43 blocks tall, 20x20 footprint. Three visual zones:
- Zone 1: Stone base arcade (2 floors, 9 blocks)
- Zone 2: White brick facade with windows (6 floors, 21 blocks)
- Zone 3: Mansard roof tapering from 20x20 to ~10x10 (3 floors, 13 blocks)
Plus 2 corner towers rising 5 blocks above the roofline.
"""

from ..engine.world import World
from ..engine.palette import WITTE_HUIS, Palette
from ..core.shapes import filled_rectangle, rectangle_shell
from ..core.extrusion import extrude_box
from ..patterns.window_grid import window_grid
from ..patterns.sloped_roof import mansard_taper, conical_cap
from ..patterns.floor_stack import floor_stack
from ..patterns.facade_detail import column_arcade, cornice_band, dormer


def build(world: World, palette: Palette = None, origin_x: int = 0, origin_z: int = 0) -> None:
    """Build the Witte Huis at 1:1 scale (43 blocks)."""
    p = palette or WITTE_HUIS
    W, D = 20, 20  # Footprint

    # ===== ZONE 1: Stone base (Y 0-8, 9 blocks) =====
    base_shell = extrude_box(W, 9, D, origin_x - W // 2, 0, origin_z - D // 2,
                             hollow=True, wall_thickness=1)
    world.set_blocks(base_shell, p["stone_base"])

    # Ground floor arcade on south face (Z-)
    arcade_south = column_arcade(
        width=W, height=5, num_columns=5,
        start_x=origin_x - W // 2, start_y=0, start_z=origin_z - D // 2,
        face="z", column_block=p["columns"], arch_block=p["stone_base"],
    )
    world.set_block_dict(arcade_south)

    # Cornice between zone 1 and zone 2
    cornice_1 = cornice_band(
        W, D, start_x=origin_x, start_y=9, start_z=origin_z,
        block=p["cornice"], extend_out=1,
    )
    world.set_block_dict(cornice_1)

    # Interior floors for base
    base_footprint = filled_rectangle(W - 2, D - 2)
    base_floors = floor_stack(base_footprint, floor_interval=5, num_floors=2,
                              start_y=0, floor_block=p["floor"])
    world.set_block_dict(base_floors)

    # ===== ZONE 2: White brick facade (Y 9-29, 21 blocks, 6 floors) =====
    # Four walls with windows
    for face_dir, sx, sz, fw in [
        ("z", origin_x - W // 2, origin_z - D // 2, W),    # South
        ("z", origin_x - W // 2, origin_z + D // 2, W),    # North
        ("x", origin_x - W // 2, origin_z - D // 2, D),    # West
        ("x", origin_x + W // 2, origin_z - D // 2, D),    # East
    ]:
        wall = window_grid(
            wall_width=fw, wall_height=21,
            window_width=1, window_height=2,
            h_spacing=3, v_spacing=4,
            margin_x=2, margin_y=1,
            start_x=sx, start_y=9, start_z=sz,
            face=face_dir,
            wall_block=p["white_facade"], window_block=p["windows"],
        )
        world.set_block_dict(wall)

    # Interior floors for zone 2
    zone2_footprint = filled_rectangle(W - 2, D - 2)
    zone2_floors = floor_stack(zone2_footprint, floor_interval=4, num_floors=6,
                               start_y=9, floor_block=p["floor"])
    world.set_block_dict(zone2_floors)

    # Cornice between zone 2 and zone 3
    cornice_2 = cornice_band(
        W, D, start_x=origin_x, start_y=30, start_z=origin_z,
        block=p["cornice"], extend_out=1,
    )
    world.set_block_dict(cornice_2)

    # ===== ZONE 3: Mansard roof (Y 30-42, 13 blocks) =====
    mansard = mansard_taper(
        base_width=W, base_depth=D, height=13,
        taper_per_layer=0.4,
        start_x=origin_x, start_y=30, start_z=origin_z,
        wall_block=p["mansard_roof"],
    )
    world.set_block_dict(mansard)

    # Dormers on each face (2 per face)
    for face_dir, dx_list, dz, face in [
        ("z", [-5, 5], origin_z - D // 2 - 1, "z"),  # South
        ("z", [-5, 5], origin_z + D // 2 + 1, "z"),   # North
    ]:
        for dx in dx_list:
            d = dormer(
                width=3, height=3, depth=2,
                start_x=origin_x + dx, start_y=33, start_z=dz,
                face=face, wall_block=p["white_facade"],
                roof_block=p["mansard_roof"], window_block=p["windows"],
            )
            world.set_block_dict(d)

    # ===== Corner towers (2 on south corners, Y 0-48) =====
    tower_positions = [
        (origin_x - W // 2 + 1, origin_z - D // 2 + 1),  # SW corner
        (origin_x + W // 2 - 2, origin_z - D // 2 + 1),  # SE corner
    ]

    for tx, tz in tower_positions:
        # Tower shaft extends above roof
        for y in range(30, 44):
            for dx in range(3):
                for dz in range(3):
                    world.set_block(tx + dx, y, tz + dz, p["white_facade"])

        # Conical cap
        cap = conical_cap(
            base_diameter=3, height=5,
            start_x=tx + 1, start_y=44, start_z=tz + 1,
            block=p["tower_cap"],
        )
        world.set_block_dict(cap)

        # Finial
        world.set_block(tx + 1, 49, tz + 1, p["iron_details"])
