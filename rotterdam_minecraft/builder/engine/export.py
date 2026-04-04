"""Export world to various Minecraft-compatible formats.

Supports:
- .mcfunction (vanilla /setblock commands)
- WorldEdit script (//pos1, //pos2, //set sequences)
- CSV (debugging)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
from .world import World


def to_mcfunction(
    world: World,
    filename: str,
    origin_x: int = 0,
    origin_y: int = 64,
    origin_z: int = 0,
    batch_size: int = 50000,
) -> list[str]:
    """Export to .mcfunction file(s) with /setblock commands.

    For builds over batch_size blocks, splits into multiple files
    (Minecraft has command limits per function).

    Args:
        world: The World to export.
        filename: Output path (without extension, will add .mcfunction).
        origin_x/y/z: World offset (Y=64 is typical ground level).
        batch_size: Max commands per file.

    Returns:
        List of filenames created.
    """
    blocks = world.blocks()
    # Sort by Y then X then Z for consistent placement order
    sorted_blocks = sorted(blocks.items(), key=lambda p: (p[0][1], p[0][0], p[0][2]))

    files_created = []
    file_index = 0

    for batch_start in range(0, len(sorted_blocks), batch_size):
        batch = sorted_blocks[batch_start:batch_start + batch_size]

        if len(sorted_blocks) <= batch_size:
            out_path = f"{filename}.mcfunction"
        else:
            out_path = f"{filename}_part{file_index}.mcfunction"

        lines = []
        for (x, y, z), block in batch:
            wx = x + origin_x
            wy = y + origin_y
            wz = z + origin_z
            lines.append(f"setblock {wx} {wy} {wz} {block} replace")

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\n".join(lines))

        files_created.append(out_path)
        file_index += 1

    return files_created


def to_worldedit(
    world: World,
    filename: str,
    origin_x: int = 0,
    origin_y: int = 64,
    origin_z: int = 0,
) -> str:
    """Export to a WorldEdit macro script.

    Groups blocks by type and uses //set for efficiency where possible.
    Falls back to individual //pos1 + //pos2 + //set for scattered blocks.
    """
    blocks = world.blocks()
    out_path = f"{filename}_we.txt"

    # Group by block type
    by_type: dict[str, list] = {}
    for (x, y, z), block in blocks.items():
        by_type.setdefault(block, []).append((x + origin_x, y + origin_y, z + origin_z))

    lines = ["// WorldEdit script - paste into chat or run as macro"]
    for block_type, positions in by_type.items():
        lines.append(f"// {block_type}: {len(positions)} blocks")
        for wx, wy, wz in sorted(positions):
            lines.append(f"//pos1 {wx},{wy},{wz}")
            lines.append(f"//pos2 {wx},{wy},{wz}")
            lines.append(f"//set {block_type}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    return out_path


def to_csv(
    world: World,
    filename: str,
    origin_x: int = 0,
    origin_y: int = 64,
    origin_z: int = 0,
) -> str:
    """Export to CSV for debugging/visualization."""
    blocks = world.blocks()
    out_path = f"{filename}.csv"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("x,y,z,block\n")
        for (x, y, z), block in sorted(blocks.items()):
            f.write(f"{x + origin_x},{y + origin_y},{z + origin_z},{block}\n")

    return out_path


def print_summary(world: World, name: str = "Build") -> None:
    """Print a summary of the world contents."""
    dims = world.dimensions()
    bounds = world.bounds()
    materials = world.material_list()

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Total blocks: {world.block_count():,}")
    print(f"  Dimensions:   {dims[0]} x {dims[1]} x {dims[2]}")
    print(f"  Bounds:       {bounds[0]} to {bounds[1]}")
    print(f"\n  Materials:")
    for block, count in sorted(materials.items(), key=lambda x: -x[1]):
        short_name = block.replace("minecraft:", "")
        print(f"    {short_name:40s} {count:>8,}")
    print(f"{'='*60}\n")
