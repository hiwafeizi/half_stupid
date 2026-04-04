"""CLI entry point for building Rotterdam Minecraft buildings.

Usage:
    python -m builder.build_all                    # Build all 10
    python -m builder.build_all euromast            # Build one
    python -m builder.build_all euromast witte_huis # Build multiple
    python -m builder.build_all --list              # List available buildings
    python -m builder.build_all --format csv        # Export format
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from builder.engine.world import World
from builder.engine.export import to_mcfunction, to_csv, print_summary
from builder.buildings import ALL_BUILDINGS


def main():
    args = sys.argv[1:]

    # Parse options
    export_format = "mcfunction"
    origin_y = 64
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")

    # Filter options from building names
    buildings_to_build = []
    i = 0
    while i < len(args):
        if args[i] == "--list":
            print("Available buildings:")
            for name in ALL_BUILDINGS:
                print(f"  {name}")
            return
        elif args[i] == "--format":
            i += 1
            export_format = args[i]
        elif args[i] == "--origin-y":
            i += 1
            origin_y = int(args[i])
        elif args[i] == "--output":
            i += 1
            output_dir = args[i]
        elif args[i] == "--help" or args[i] == "-h":
            print(__doc__)
            return
        else:
            buildings_to_build.append(args[i])
        i += 1

    # Default: build all
    if not buildings_to_build:
        buildings_to_build = list(ALL_BUILDINGS.keys())

    # Validate names
    for name in buildings_to_build:
        if name not in ALL_BUILDINGS:
            print(f"Unknown building: '{name}'")
            print(f"Available: {', '.join(ALL_BUILDINGS.keys())}")
            sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Build each requested building
    spacing = 300  # Space between buildings when building multiple
    origin_x = 0

    for name in buildings_to_build:
        print(f"\nBuilding {name}...")
        world = World()

        build_fn = ALL_BUILDINGS[name]
        build_fn(world, origin_x=origin_x, origin_z=0)

        # Print summary
        print_summary(world, name)

        # Export
        out_path = os.path.join(output_dir, name)
        if export_format == "csv":
            f = to_csv(world, out_path, origin_y=origin_y)
            print(f"  Exported to {f}")
        else:
            files = to_mcfunction(world, out_path, origin_y=origin_y)
            for f in files:
                print(f"  Exported to {f}")

        origin_x += spacing

    print(f"\nDone! {len(buildings_to_build)} building(s) exported to {output_dir}/")


if __name__ == "__main__":
    main()
