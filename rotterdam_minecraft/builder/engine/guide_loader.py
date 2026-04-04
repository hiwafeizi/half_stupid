"""Load and parse build_guide.json files.

Extracts structured data that building scripts consume:
dimensions, layer profiles, circle templates, block palettes.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Optional


def load_guide(building_folder: str) -> dict:
    """Load a build_guide.json from a building folder.

    Args:
        building_folder: Path to the building's build_guides folder.
            e.g., "c:/github/half_stupid/rotterdam_minecraft/build_guides/04_euromast"

    Returns:
        Parsed JSON as a dict.
    """
    path = Path(building_folder) / "build_guide.json"
    if not path.exists():
        raise FileNotFoundError(f"No build_guide.json in {building_folder}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_research(building_folder: str) -> dict:
    """Load a building.json from a research folder.

    Args:
        building_folder: Path to the building's research folder.

    Returns:
        Parsed JSON as a dict.
    """
    path = Path(building_folder) / "building.json"
    if not path.exists():
        raise FileNotFoundError(f"No building.json in {building_folder}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dimensions(guide: dict, key: str, default: Any = 0) -> Any:
    """Safely extract a dimension value from nested guide data."""
    # Try common locations
    for section_key in ["dimensions", "pylon_construction", "shaft_construction",
                        "main_body", "tower_layout", "arch_construction"]:
        section = guide.get(section_key, {})
        if isinstance(section, dict) and key in section:
            return section[key]

    # Try top-level
    return guide.get(key, default)


BASE_DIR = Path("c:/github/half_stupid/rotterdam_minecraft")
GUIDES_DIR = BASE_DIR / "build_guides"
RESEARCH_DIR = BASE_DIR / "research"

BUILDING_FOLDERS = {
    "erasmus_bridge": "01_erasmus_bridge",
    "cube_houses": "02_cube_houses",
    "markthal": "03_markthal",
    "euromast": "04_euromast",
    "rotterdam_centraal": "05_rotterdam_centraal",
    "depot_boijmans": "06_depot_boijmans",
    "de_rotterdam": "07_de_rotterdam",
    "hotel_new_york": "08_hotel_new_york",
    "witte_huis": "09_witte_huis",
    "van_nelle_factory": "10_van_nelle_factory",
}


def load_building_guide(name: str) -> dict:
    """Load guide by building short name.

    Args:
        name: One of: erasmus_bridge, cube_houses, markthal, euromast,
              rotterdam_centraal, depot_boijmans, de_rotterdam,
              hotel_new_york, witte_huis, van_nelle_factory
    """
    folder = BUILDING_FOLDERS.get(name)
    if not folder:
        raise ValueError(f"Unknown building: {name}. Options: {list(BUILDING_FOLDERS.keys())}")
    return load_guide(str(GUIDES_DIR / folder))
