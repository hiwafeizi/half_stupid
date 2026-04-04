"""3D voxel viewer for Rotterdam Minecraft buildings.

Uses matplotlib's voxel rendering to visualize builds.

Usage:
    python -m builder.viewer euromast
    python -m builder.viewer witte_huis
    python -m builder.viewer depot_boijmans
    python -m builder.viewer --list
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from builder.engine.world import World
from builder.buildings import ALL_BUILDINGS

# Block color mapping (Minecraft block ID -> RGBA)
BLOCK_COLORS = {
    # Concrete
    "minecraft:white_concrete": (0.95, 0.95, 0.95, 0.9),
    "minecraft:light_gray_concrete": (0.7, 0.7, 0.7, 0.9),
    "minecraft:gray_concrete": (0.5, 0.5, 0.5, 0.9),
    "minecraft:black_concrete": (0.15, 0.15, 0.15, 0.9),
    "minecraft:yellow_concrete": (0.95, 0.85, 0.2, 0.9),
    "minecraft:red_concrete": (0.8, 0.2, 0.2, 0.9),
    "minecraft:light_blue_concrete": (0.55, 0.75, 0.95, 0.9),
    "minecraft:blue_concrete": (0.2, 0.3, 0.8, 0.9),
    "minecraft:brown_concrete": (0.5, 0.35, 0.2, 0.9),
    "minecraft:green_concrete": (0.3, 0.6, 0.3, 0.9),
    "minecraft:orange_concrete": (0.9, 0.5, 0.1, 0.9),
    "minecraft:cyan_concrete": (0.2, 0.6, 0.6, 0.9),

    # Stone
    "minecraft:stone": (0.55, 0.55, 0.55, 0.9),
    "minecraft:smooth_stone": (0.6, 0.6, 0.6, 0.9),
    "minecraft:smooth_stone_slab": (0.6, 0.6, 0.6, 0.9),
    "minecraft:stone_bricks": (0.55, 0.55, 0.55, 0.9),
    "minecraft:polished_andesite": (0.5, 0.5, 0.5, 0.9),
    "minecraft:deepslate": (0.3, 0.3, 0.35, 0.9),
    "minecraft:deepslate_tiles": (0.3, 0.3, 0.35, 0.9),

    # Metal
    "minecraft:iron_block": (0.8, 0.8, 0.82, 0.9),
    "minecraft:iron_bars": (0.5, 0.5, 0.52, 0.7),
    "minecraft:gold_block": (0.95, 0.8, 0.2, 0.9),
    "minecraft:chain": (0.4, 0.4, 0.4, 0.7),
    "minecraft:lightning_rod": (0.75, 0.55, 0.3, 0.9),

    # Copper
    "minecraft:oxidized_copper": (0.4, 0.7, 0.6, 0.9),
    "minecraft:oxidized_cut_copper": (0.4, 0.7, 0.6, 0.9),

    # Glass
    "minecraft:glass": (0.7, 0.85, 0.95, 0.3),
    "minecraft:glass_pane": (0.7, 0.85, 0.95, 0.3),
    "minecraft:light_blue_stained_glass": (0.5, 0.7, 0.9, 0.3),
    "minecraft:light_blue_stained_glass_pane": (0.5, 0.7, 0.9, 0.3),
    "minecraft:blue_stained_glass_pane": (0.2, 0.3, 0.8, 0.3),
    "minecraft:light_gray_stained_glass_pane": (0.6, 0.6, 0.6, 0.3),

    # Wood
    "minecraft:oak_planks": (0.7, 0.55, 0.35, 0.9),
    "minecraft:oak_slab": (0.7, 0.55, 0.35, 0.9),
    "minecraft:spruce_planks": (0.45, 0.35, 0.2, 0.9),
    "minecraft:birch_planks": (0.85, 0.8, 0.6, 0.9),
    "minecraft:stripped_oak_log": (0.7, 0.6, 0.4, 0.9),
    "minecraft:birch_log": (0.9, 0.88, 0.85, 0.9),

    # Brick / Terracotta
    "minecraft:bricks": (0.65, 0.35, 0.25, 0.9),
    "minecraft:red_terracotta": (0.6, 0.3, 0.2, 0.9),
    "minecraft:yellow_terracotta": (0.75, 0.65, 0.3, 0.9),
    "minecraft:white_terracotta": (0.85, 0.82, 0.78, 0.9),

    # Glazed terracotta
    "minecraft:orange_glazed_terracotta": (0.9, 0.5, 0.15, 0.9),
    "minecraft:green_glazed_terracotta": (0.3, 0.65, 0.3, 0.9),
    "minecraft:blue_glazed_terracotta": (0.2, 0.35, 0.75, 0.9),
    "minecraft:red_glazed_terracotta": (0.8, 0.2, 0.2, 0.9),
    "minecraft:yellow_glazed_terracotta": (0.9, 0.85, 0.2, 0.9),
    "minecraft:cyan_glazed_terracotta": (0.2, 0.6, 0.6, 0.9),

    # Quartz
    "minecraft:quartz_block": (0.92, 0.9, 0.88, 0.9),
    "minecraft:quartz_pillar": (0.92, 0.9, 0.88, 0.9),
    "minecraft:quartz_stairs": (0.92, 0.9, 0.88, 0.9),

    # Nature
    "minecraft:grass_block": (0.3, 0.7, 0.2, 0.9),
    "minecraft:water": (0.2, 0.4, 0.8, 0.5),
    "minecraft:birch_leaves": (0.4, 0.65, 0.3, 0.6),
    "minecraft:oak_leaves": (0.2, 0.55, 0.15, 0.6),
    "minecraft:spruce_leaves": (0.15, 0.4, 0.15, 0.6),

    # Prismarine
    "minecraft:dark_prismarine": (0.2, 0.4, 0.35, 0.9),

    # Lighting
    "minecraft:sea_lantern": (0.8, 0.9, 0.95, 0.9),
    "minecraft:glowstone": (0.9, 0.8, 0.5, 0.9),

    # Misc
    "minecraft:end_rod": (0.9, 0.85, 0.8, 0.5),
    "minecraft:sandstone": (0.85, 0.8, 0.6, 0.9),
    "minecraft:cut_sandstone": (0.85, 0.8, 0.6, 0.9),
}

DEFAULT_COLOR = (0.6, 0.6, 0.6, 0.8)


def downsample_world(world: World, factor: int = 1):
    """Downsample a world by a factor (combine factor³ blocks into 1).

    For large builds, rendering every block is too slow.
    This picks the most common block in each factor³ cube.
    """
    if factor <= 1:
        return world

    from collections import Counter
    new_world = World()
    blocks = world.blocks()

    # Group blocks by downsampled position
    groups = {}
    for (x, y, z), block in blocks.items():
        key = (x // factor, y // factor, z // factor)
        if key not in groups:
            groups[key] = []
        groups[key].append(block)

    # Pick most common block in each group
    for (x, y, z), block_list in groups.items():
        most_common = Counter(block_list).most_common(1)[0][0]
        new_world.set_block(x, y, z, most_common)

    return new_world


def render_world(world: World, title: str = "Building", max_blocks: int = 15000):
    """Render a World using matplotlib 3D voxels."""
    import matplotlib.pyplot as plt

    blocks = world.blocks()
    total = len(blocks)

    # Auto-downsample if too many blocks
    ds_factor = 1
    while total / (ds_factor ** 3) > max_blocks:
        ds_factor += 1

    if ds_factor > 1:
        print(f"  Downsampling {ds_factor}x ({total:,} -> ~{total // ds_factor**3:,} blocks)")
        world = downsample_world(world, ds_factor)
        blocks = world.blocks()

    (min_x, min_y, min_z), (max_x, max_y, max_z) = world.bounds()
    sx = max_x - min_x + 1
    sy = max_y - min_y + 1
    sz = max_z - min_z + 1

    print(f"  Rendering {len(blocks):,} blocks ({sx}x{sy}x{sz})...")

    # Build voxel array and color array
    voxels = np.zeros((sx, sy, sz), dtype=bool)
    colors = np.zeros((sx, sy, sz, 4))

    for (x, y, z), block in blocks.items():
        ix = x - min_x
        iy = y - min_y
        iz = z - min_z
        voxels[ix, iy, iz] = True
        colors[ix, iy, iz] = BLOCK_COLORS.get(block, DEFAULT_COLOR)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # For large builds, only render outer surface (hide interior)
    if len(blocks) > 5000:
        # Remove interior blocks (surrounded on all 6 sides)
        for ix in range(1, sx - 1):
            for iy in range(1, sy - 1):
                for iz in range(1, sz - 1):
                    if voxels[ix, iy, iz]:
                        if (voxels[ix-1, iy, iz] and voxels[ix+1, iy, iz] and
                            voxels[ix, iy-1, iz] and voxels[ix, iy+1, iz] and
                            voxels[ix, iy, iz-1] and voxels[ix, iy, iz+1]):
                            voxels[ix, iy, iz] = False

    visible = np.sum(voxels)
    print(f"  Visible surface blocks: {visible:,}")

    ax.voxels(voxels, facecolors=colors, edgecolors=None)

    ax.set_xlabel('X')
    ax.set_ylabel('Y (height)')
    ax.set_zlabel('Z')
    ax.set_title(f"{title}\n{total:,} blocks | {sx}x{sy}x{sz}")

    # Equal aspect ratio
    max_range = max(sx, sy, sz)
    ax.set_xlim(0, max_range)
    ax.set_ylim(0, max_range)
    ax.set_zlim(0, max_range)

    plt.tight_layout()
    plt.show()


def render_slices(world: World, title: str = "Building", num_slices: int = 6):
    """Render horizontal slices (floor plans) at different heights."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    (_, min_y, _), (_, max_y, _) = world.bounds()
    height = max_y - min_y + 1

    # Pick evenly spaced heights
    step = max(1, height // num_slices)
    heights = list(range(min_y, max_y + 1, step))[:num_slices]

    fig, axes = plt.subplots(2, (num_slices + 1) // 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, y in enumerate(heights):
        if idx >= len(axes):
            break

        ax = axes[idx]
        layer = world.slice_y(y)

        if not layer:
            ax.set_title(f"Y={y} (empty)")
            continue

        xs = [pos[0] for pos in layer]
        zs = [pos[1] for pos in layer]
        cs = [BLOCK_COLORS.get(block, DEFAULT_COLOR)[:3] for block in layer.values()]

        ax.scatter(xs, zs, c=cs, s=2, marker='s')
        ax.set_aspect('equal')
        ax.set_title(f"Y={y}")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"{title} - Horizontal Slices", fontsize=14)
    plt.tight_layout()
    plt.show()


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

    mode = "3d"  # or "slices"
    buildings = []
    for arg in args:
        if arg == "--slices":
            mode = "slices"
        elif arg in ALL_BUILDINGS:
            buildings.append(arg)
        else:
            print(f"Unknown: '{arg}'. Use --list to see options.")
            return

    if not buildings:
        print("No building specified. Use --list to see options.")
        return

    for name in buildings:
        print(f"\nBuilding {name}...")
        world = World()
        ALL_BUILDINGS[name](world, origin_x=0, origin_z=0)
        print(f"  {world.block_count():,} blocks, dimensions {world.dimensions()}")

        if mode == "slices":
            render_slices(world, title=name.replace("_", " ").title())
        else:
            render_world(world, title=name.replace("_", " ").title())


if __name__ == "__main__":
    main()
