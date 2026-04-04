"""Van Nelle Factory builder — stepped glass cascade, UNESCO World Heritage.

230 blocks long, 19 deep. Three sections: Tobacco (32), Coffee (20), Tea (12).
Plus conveyor bridges, curved office, stair towers, rooftop tearoom.
"""

from ..engine.world import World
from ..engine.palette import VAN_NELLE, Palette
from ..core.shapes import filled_rectangle, filled_circle, circle_outline
from ..core.extrusion import extrude_box, extrude_cylinder
from ..patterns.curtain_wall import curtain_wall
from ..patterns.floor_stack import floor_stack
from ..patterns.cables import sloped_bridge
from ..core.curves import arc_points
import math


def _factory_section(
    world: World, p,
    start_x: int, start_z: int,
    length: int, depth: int, height: int,
    floors: int, column_spacing: int = 6,
):
    """Build one factory section with glass curtain walls and mushroom columns."""

    # Glass curtain walls (front and back, full height)
    for face_z, face_dir in [(start_z, "z"), (start_z + depth - 1, "z")]:
        cw = curtain_wall(
            width=length, height=height,
            mullion_spacing=column_spacing, spandrel_spacing=4,
            start_x=start_x, start_y=0, start_z=face_z,
            face="z",
            glass_block=p["glass_walls"],
            mullion_block=p["columns"],
            spandrel_block=p["floor_slabs"],
            column_setback=1,
            column_block=p["columns"],
            column_spacing=column_spacing,
        )
        world.set_block_dict(cw)

    # Side walls (glass)
    for face_x in [start_x, start_x + length - 1]:
        cw = curtain_wall(
            width=depth, height=height,
            mullion_spacing=column_spacing, spandrel_spacing=4,
            start_x=face_x, start_y=0, start_z=start_z,
            face="x",
            glass_block=p["glass_walls"],
            mullion_block=p["columns"],
            spandrel_block=p["floor_slabs"],
        )
        world.set_block_dict(cw)

    # Mushroom columns (interior grid)
    for x in range(start_x + column_spacing, start_x + length - 1, column_spacing):
        for z in range(start_z + 3, start_z + depth - 2, column_spacing):
            for y in range(0, height):
                world.set_block(x, y, z, p["columns"])
            # Mushroom capital (slab extending 1 block out) at each floor
            for floor_y in range(4, height, 4):
                for dx in range(-1, 2):
                    for dz in range(-1, 2):
                        world.set_block(x + dx, floor_y - 1, z + dz, p["mushroom_capitals"])

    # Floor slabs at intervals
    floor_fp = filled_rectangle(length - 2, depth - 2)
    floors_dict = floor_stack(
        floor_fp, floor_interval=4, num_floors=floors,
        start_y=0, floor_block=p["floor_slabs"],
        center_x=start_x + length // 2,
        center_z=start_z + depth // 2,
    )
    world.set_block_dict(floors_dict)

    # Flat roof
    for x in range(start_x, start_x + length):
        for z in range(start_z, start_z + depth):
            world.set_block(x, height, z, p["roof"])


def build(world: World, palette: Palette = None, origin_x: int = 0, origin_z: int = 0) -> None:
    """Build Van Nelle Factory at 1:1 scale (230 blocks long)."""
    p = palette or VAN_NELLE

    DEPTH = 19
    COL_SPACING = 6
    ROAD_WIDTH = 13  # Between factory and dispatch hall

    # Sections arranged along X axis (south to north = tallest to shortest)
    tobacco_x = origin_x
    tobacco_len = 130
    tobacco_h = 32

    coffee_x = tobacco_x + tobacco_len
    coffee_len = 65
    coffee_h = 20

    tea_x = coffee_x + coffee_len
    tea_len = 35
    tea_h = 12

    # ===== 1. Tobacco building (8 floors, 32 blocks) =====
    _factory_section(world, p, tobacco_x, origin_z, tobacco_len, DEPTH, tobacco_h, 8, COL_SPACING)

    # Rooftop circular tearoom
    tearoom_cx = tobacco_x + tobacco_len // 2
    tearoom_cz = origin_z + DEPTH // 2
    tearoom_r = 4
    tearoom_h = 5
    tearoom = extrude_cylinder(tearoom_r, tearoom_h, start_y=tobacco_h + 1, hollow=True, wall_thickness=1)
    for x, y, z in tearoom:
        world.set_block(tearoom_cx + x, y, tearoom_cz + z, p["tearoom"])
    # Tearoom roof
    for x, z in filled_circle(tearoom_r):
        world.set_block(tearoom_cx + x, tobacco_h + tearoom_h + 1, tearoom_cz + z, p["roof"])

    # ===== 2. Coffee building (5 floors, 20 blocks) =====
    _factory_section(world, p, coffee_x, origin_z, coffee_len, DEPTH, coffee_h, 5, COL_SPACING)

    # ===== 3. Tea building (3 floors, 12 blocks) =====
    _factory_section(world, p, tea_x, origin_z, tea_len, DEPTH, tea_h, 3, COL_SPACING)

    # ===== 4. Storage/dispatch hall (across internal road) =====
    dispatch_z = origin_z + DEPTH + ROAD_WIDTH
    dispatch_len = 170
    dispatch_h = 9
    _factory_section(world, p, tobacco_x + 10, dispatch_z, dispatch_len, 17, dispatch_h, 2, COL_SPACING)

    # Internal road surface
    for x in range(tobacco_x, tobacco_x + tobacco_len + coffee_len + tea_len):
        for z in range(origin_z + DEPTH, dispatch_z):
            world.set_block(x, 0, z, p["floor_slabs"])

    # ===== 5. Conveyor bridges (3 bridges connecting factory to dispatch) =====
    bridge_positions = [
        (tobacco_x + 40, tobacco_h - 8),   # From tobacco
        (tobacco_x + 90, tobacco_h - 12),   # From tobacco
        (coffee_x + 30, coffee_h - 4),      # From coffee
    ]

    for bx, by in bridge_positions:
        bridge = sloped_bridge(
            start_x=bx, start_y=by, start_z=origin_z + DEPTH,
            end_x=bx, end_y=by - 4, end_z=dispatch_z,
            width=3, height=3,
            wall_block=p["conveyor_glass"],
            floor_block=p["conveyor_floor"],
            frame_block=p["conveyor_frame"],
        )
        world.set_block_dict(bridge)

    # ===== 6. Stair towers (4 cylindrical, at section junctions) =====
    stair_positions = [
        (tobacco_x + 2, origin_z + DEPTH // 2, tobacco_h),
        (tobacco_x + tobacco_len - 2, origin_z + DEPTH // 2, tobacco_h),
        (coffee_x + coffee_len - 2, origin_z + DEPTH // 2, coffee_h),
        (tea_x + tea_len - 2, origin_z + DEPTH // 2, tea_h),
    ]

    for sx, sz, sh in stair_positions:
        stair = extrude_cylinder(3.5, sh, start_y=0, hollow=True, wall_thickness=1)
        for x, y, z in stair:
            world.set_block(sx + x, y, sz + z, p["stair_tower"])
        # Glass strips (windows)
        for y in range(0, sh):
            world.set_block(sx, y, sz + 3, p["glass_walls"])
            world.set_block(sx, y, sz - 3, p["glass_walls"])

    # ===== 7. Curved office building (front) =====
    curve_cx = tobacco_x + 30
    curve_cz = origin_z - 10
    curve_r = 35
    curve_h = 12

    # Arc from ~160° to ~200° (40 degree segment)
    for y in range(curve_h):
        arc = arc_points(curve_r, 160, 200)
        for x, z in arc:
            world.set_block(curve_cx + x, y, curve_cz + z, p["curved_office"])
        # Floor slabs
        if y % 4 == 0:
            inner_arc = arc_points(curve_r - 9, 160, 200)
            # Fill between inner and outer arcs
            for x, z in arc:
                for ix, iz in inner_arc:
                    if abs(x - ix) <= 1 and abs(z - iz) <= 1:
                        world.set_block(curve_cx + x, y, curve_cz + z, p["floor_slabs"])

    # ===== 8. Canal (along the front) =====
    for x in range(tobacco_x - 5, tea_x + tea_len + 5):
        for z in range(origin_z - 20, origin_z - 12):
            world.set_block(x, -1, z, "minecraft:water")
