"""Floor slab insertion at regular intervals within a volume.

Used by every multi-story building. Supports constant footprints,
shrinking footprints (mansard), growing footprints (bowl), and
footprints with voids (atriums).
"""

from __future__ import annotations
from typing import Dict, Set, Tuple, Callable, Optional
from ..core.blocks import Block
from ..core.shapes import filled_rectangle, filled_circle

Coord3D = Tuple[int, int, int]
Coord2D = Tuple[int, int]


def floor_stack(
    footprint: Set[Coord2D],
    floor_interval: int,
    num_floors: int,
    start_y: int = 0,
    floor_block: str = Block.SMOOTH_STONE,
    void_region: Optional[Set[Coord2D]] = None,
    center_x: int = 0,
    center_z: int = 0,
) -> Dict[Coord3D, str]:
    """Insert horizontal floor slabs at regular intervals.

    Args:
        footprint: 2D footprint (x, z) of each floor.
        floor_interval: Blocks between floors.
        num_floors: How many floors to create.
        start_y: Y of first floor slab.
        floor_block: Block type for floor slabs.
        void_region: Optional (x, z) set to subtract (atrium/elevator void).
        center_x/z: Translation offset for the footprint.
    """
    blocks: Dict[Coord3D, str] = {}

    active = footprint
    if void_region:
        active = footprint - void_region

    for floor_num in range(num_floors):
        y = start_y + floor_num * floor_interval
        for x, z in active:
            blocks[(x + center_x, y, z + center_z)] = floor_block

    return blocks


def floor_stack_circular(
    diameter_at_floor: Callable[[int], float],
    floor_interval: int,
    num_floors: int,
    start_y: int = 0,
    floor_block: str = Block.SMOOTH_STONE,
    wall_thickness: int = 2,
    void_width: int = 0,
    void_depth: int = 0,
) -> Dict[Coord3D, str]:
    """Insert circular floor slabs with variable diameter per floor.

    Used by Depot Boijmans where each floor has a different diameter.

    Args:
        diameter_at_floor: Function taking floor_index -> diameter.
        floor_interval: Blocks between floors.
        num_floors: Number of floors.
        start_y: Y of first floor.
        floor_block: Block type.
        wall_thickness: Wall thickness (floors extend wall_thickness inward).
        void_width/depth: Rectangular atrium void at center.
    """
    blocks: Dict[Coord3D, str] = {}

    void = set()
    if void_width > 0 and void_depth > 0:
        void = filled_rectangle(void_width, void_depth)

    for floor_num in range(num_floors):
        y = start_y + floor_num * floor_interval
        d = diameter_at_floor(floor_num)
        footprint = filled_circle(d / 2) - void
        for x, z in footprint:
            blocks[(x, y, z)] = floor_block

    return blocks
