"""Launch Rotterdam buildings in Minecraft via Malmo.

Usage:
    python build_rotterdam.py --all              # All 10 buildings
    python build_rotterdam.py euromast            # Just one
    python build_rotterdam.py euromast witte_huis # Multiple
    python build_rotterdam.py --list             # List available

Prerequisites:
    1. Run: C:\\Users\\hiwa\\Malmo_Python3.7\\Minecraft\\launchClient.bat
    2. Wait for DORMANT state
    3. conda activate malmo
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from builder.engine.world import World
from builder.engine.malmo_export import run_in_malmo
from builder.buildings import ALL_BUILDINGS

GROUND_Y = 4


def main():
    args = sys.argv[1:]

    if not args or "--help" in args or "-h" in args:
        print(__doc__)
        return

    if "--list" in args:
        print("Available buildings:")
        for name in ALL_BUILDINGS:
            print(f"  {name}")
        return

    build_all = "--all" in args
    spacing = 10

    if build_all:
        building_names = list(ALL_BUILDINGS.keys())
    else:
        building_names = [a for a in args if a in ALL_BUILDINGS]
        if not building_names:
            unknown = [a for a in args if not a.startswith("--")]
            if unknown:
                print(f"Unknown: {unknown}")
            print(f"Available: {', '.join(ALL_BUILDINGS.keys())}")
            return

    # Generate all building worlds
    print(f"Generating {len(building_names)} building(s)...")
    worlds = []
    for name in building_names:
        print(f"  {name}...", end=" ")
        w = World()
        ALL_BUILDINGS[name](w, origin_x=0, origin_z=0)
        print(f"{w.block_count():,} blocks")
        worlds.append((name, w))

    total = sum(w.block_count() for _, w in worlds)
    print(f"Total: {total:,} blocks\n")

    # Run in Malmo
    try:
        run_in_malmo(worlds, origin_y=GROUND_Y, spacing=spacing)
    except ImportError:
        print("MalmoPython not available!")
        print("Run: conda activate malmo")
    except RuntimeError as e:
        print(f"Failed: {e}")
        print("Make sure launchClient.bat is running and in DORMANT state.")

    print("\nDone!")


if __name__ == "__main__":
    main()
