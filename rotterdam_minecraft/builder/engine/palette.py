"""Named palette system for semantic block references.

Each building defines a palette mapping role names to Minecraft block IDs.
Building scripts reference palette["facade"] instead of hardcoded block names,
making material swaps trivial.
"""

from __future__ import annotations
from typing import Dict, Optional
from ..core.blocks import Block


class Palette:
    """Named block palette for a building."""

    def __init__(self, mapping: Dict[str, str]):
        self._map = mapping

    def __getitem__(self, key: str) -> str:
        return self._map[key]

    def get(self, key: str, default: Optional[str] = None) -> str:
        return self._map.get(key, default or Block.STONE)

    def update(self, overrides: Dict[str, str]) -> None:
        """Apply overrides (e.g., user wants different colors)."""
        self._map.update(overrides)

    def items(self):
        return self._map.items()


# === Pre-built palettes for each Rotterdam building ===

ERASMUS_BRIDGE = Palette({
    "pylon": Block.LIGHT_BLUE_CONCRETE,
    "deck_surface": Block.GRAY_CONCRETE,
    "deck_sides": Block.LIGHT_GRAY_CONCRETE,
    "cables": Block.END_ROD,
    "railings": Block.IRON_BARS,
    "piers": Block.STONE,
    "tram_tracks": Block.IRON_BARS,
    "water": Block.WATER,
})

CUBE_HOUSES = Palette({
    "cube_walls": Block.YELLOW_CONCRETE,
    "cube_trim": Block.YELLOW_TERRACOTTA,
    "pylon": Block.LIGHT_GRAY_CONCRETE,
    "windows": Block.GLASS_PANE,
    "interior_floor": Block.OAK_PLANKS,
    "interior_walls": Block.WHITE_CONCRETE,
    "walkway_floor": Block.SMOOTH_STONE,
})

MARKTHAL = Palette({
    "exterior": Block.LIGHT_GRAY_CONCRETE,
    "exterior_accent": Block.STONE,
    "glass_ends": Block.GLASS_PANE,
    "cable_net": Block.IRON_BARS,
    "market_floor": Block.POLISHED_ANDESITE,
    "apartment_walls": Block.WHITE_CONCRETE,
    "apartment_floor": Block.OAK_PLANKS,
    "mural_warm": Block.ORANGE_GLAZED_TERRACOTTA,
    "mural_green": Block.GREEN_GLAZED_TERRACOTTA,
    "mural_blue": Block.BLUE_GLAZED_TERRACOTTA,
    "mural_red": Block.RED_GLAZED_TERRACOTTA,
    "mural_yellow": Block.YELLOW_GLAZED_TERRACOTTA,
})

EUROMAST = Palette({
    "shaft": Block.LIGHT_GRAY_CONCRETE,
    "shaft_interior": Block.WHITE_CONCRETE,
    "crows_nest_top": Block.GRAY_CONCRETE,
    "crows_nest_underside": Block.DEEPSLATE,
    "crows_nest_windows": Block.GLASS_PANE,
    "space_tower": Block.IRON_BARS,
    "euroscoop": Block.GLASS_PANE,
    "base": Block.LIGHT_GRAY_CONCRETE,
    "antenna": Block.LIGHTNING_ROD,
})

ROTTERDAM_CENTRAAL = Palette({
    "canopy": Block.IRON_BLOCK,
    "glass_facade": Block.GLASS_PANE,
    "wooden_ceiling": Block.STRIPPED_OAK_LOG,
    "wooden_accent": Block.SPRUCE_PLANKS,
    "floor": Block.SMOOTH_STONE,
    "platform": Block.STONE,
    "tracks": Block.IRON_BARS,
    "clock_face": Block.WHITE_CONCRETE,
    "clock_rim": Block.BLACK_CONCRETE,
    "solar_panels": Block.BLUE_STAINED_GLASS_PANE,
    "roof_glass": Block.GLASS_PANE,
    "columns": Block.QUARTZ_PILLAR,
})

DEPOT_BOIJMANS = Palette({
    "facade_primary": Block.IRON_BLOCK,
    "facade_accent_1": Block.LIGHT_GRAY_CONCRETE,
    "facade_accent_2": Block.LIGHT_BLUE_STAINED_GLASS,
    "facade_accent_3": Block.WHITE_CONCRETE,
    "interior_wall": Block.WHITE_CONCRETE,
    "floor": Block.SMOOTH_STONE,
    "atrium_stairs": Block.QUARTZ_STAIRS,
    "atrium_glass": Block.GLASS_PANE,
    "rooftop_grass": Block.GRASS_BLOCK,
    "rooftop_path": Block.SMOOTH_STONE,
    "railing": Block.GLASS_PANE,
})

DE_ROTTERDAM = Palette({
    "glass_facade": Block.GLASS_PANE,
    "mullions": Block.IRON_BARS,
    "spandrel_office": Block.LIGHT_GRAY_CONCRETE,
    "spandrel_hotel": Block.GRAY_CONCRETE,
    "spandrel_residential": Block.WHITE_CONCRETE,
    "plinth": Block.GRAY_CONCRETE,
    "plinth_glass": Block.GLASS_PANE,
    "floor": Block.SMOOTH_STONE,
    "core": Block.LIGHT_GRAY_CONCRETE,
})

HOTEL_NEW_YORK = Palette({
    "walls": Block.BRICKS,
    "stone_details": Block.SMOOTH_STONE,
    "stone_base": Block.CUT_SANDSTONE,
    "tower_roofs": Block.OXIDIZED_COPPER,
    "main_roof": Block.DEEPSLATE_TILES,
    "windows": Block.GLASS_PANE,
    "iron_details": Block.IRON_BARS,
    "gold_lettering": Block.GOLD_BLOCK,
    "balconies": Block.IRON_BARS,
    "floor": Block.OAK_PLANKS,
})

WITTE_HUIS = Palette({
    "white_facade": Block.WHITE_CONCRETE,
    "stone_base": Block.POLISHED_ANDESITE,
    "mansard_roof": Block.DEEPSLATE_TILES,
    "columns": Block.QUARTZ_PILLAR,
    "cornice": Block.SMOOTH_STONE_SLAB,
    "windows": Block.GLASS_PANE,
    "iron_details": Block.IRON_BARS,
    "tower_cap": Block.DEEPSLATE_TILES,
    "floor": Block.OAK_PLANKS,
    "sculptures": Block.STONE_BRICKS,
})

VAN_NELLE = Palette({
    "glass_walls": Block.GLASS_PANE,
    "columns": Block.LIGHT_GRAY_CONCRETE,
    "floor_slabs": Block.SMOOTH_STONE,
    "mushroom_capitals": Block.SMOOTH_STONE_SLAB,
    "conveyor_glass": Block.GLASS_PANE,
    "conveyor_frame": Block.IRON_BARS,
    "conveyor_floor": Block.SMOOTH_STONE,
    "roof": Block.LIGHT_GRAY_CONCRETE,
    "stair_tower": Block.LIGHT_GRAY_CONCRETE,
    "tearoom": Block.GLASS_PANE,
    "curved_office": Block.GLASS_PANE,
})
