"""Export buildings to Minecraft via Malmo.

Places blocks using /setblock chat commands (no XML size limit).
Starts an empty flat world mission, then streams block placements.
"""

from __future__ import annotations
from pathlib import Path
from .world import World


# Malmo uses Minecraft 1.11.2 block names (no "minecraft:" prefix)
BLOCK_ID_MAP = {
    "minecraft:white_concrete": "quartz_block",
    "minecraft:light_gray_concrete": "clay",
    "minecraft:gray_concrete": "stone",
    "minecraft:black_concrete": "obsidian",
    "minecraft:yellow_concrete": "gold_block",
    "minecraft:red_concrete": "red_sandstone",
    "minecraft:light_blue_concrete": "packed_ice",
    "minecraft:blue_concrete": "lapis_block",
    "minecraft:brown_concrete": "soul_sand",
    "minecraft:green_concrete": "emerald_block",
    "minecraft:orange_concrete": "red_sandstone",
    "minecraft:cyan_concrete": "prismarine",
    "minecraft:stone": "stone",
    "minecraft:smooth_stone": "stone",
    "minecraft:smooth_stone_slab": "stone_slab",
    "minecraft:stone_bricks": "stonebrick",
    "minecraft:polished_andesite": "stone",
    "minecraft:deepslate": "stone",
    "minecraft:deepslate_tiles": "stone",
    "minecraft:deepslate_tile_slab": "stone_slab",
    "minecraft:iron_block": "iron_block",
    "minecraft:iron_bars": "iron_bars",
    "minecraft:gold_block": "gold_block",
    "minecraft:chain": "iron_bars",
    "minecraft:lightning_rod": "iron_bars",
    "minecraft:oxidized_copper": "prismarine",
    "minecraft:oxidized_cut_copper": "prismarine",
    "minecraft:copper_block": "gold_block",
    "minecraft:glass": "glass",
    "minecraft:glass_pane": "glass_pane",
    "minecraft:light_blue_stained_glass": "stained_glass 3",
    "minecraft:light_blue_stained_glass_pane": "stained_glass_pane 3",
    "minecraft:blue_stained_glass_pane": "stained_glass_pane 11",
    "minecraft:white_stained_glass_pane": "stained_glass_pane 0",
    "minecraft:light_gray_stained_glass_pane": "stained_glass_pane 8",
    "minecraft:oak_planks": "planks",
    "minecraft:oak_slab": "wooden_slab",
    "minecraft:oak_stairs": "oak_stairs",
    "minecraft:stripped_oak_log": "log",
    "minecraft:spruce_planks": "planks 1",
    "minecraft:birch_planks": "planks 2",
    "minecraft:birch_log": "log 2",
    "minecraft:stripped_birch_log": "log 2",
    "minecraft:bricks": "brick_block",
    "minecraft:red_terracotta": "stained_hardened_clay 14",
    "minecraft:brown_terracotta": "stained_hardened_clay 12",
    "minecraft:yellow_terracotta": "stained_hardened_clay 4",
    "minecraft:white_terracotta": "stained_hardened_clay 0",
    "minecraft:light_gray_terracotta": "stained_hardened_clay 8",
    "minecraft:cyan_terracotta": "stained_hardened_clay 9",
    "minecraft:green_terracotta": "stained_hardened_clay 13",
    "minecraft:orange_terracotta": "stained_hardened_clay 1",
    "minecraft:white_glazed_terracotta": "white_glazed_terracotta",
    "minecraft:yellow_glazed_terracotta": "yellow_glazed_terracotta",
    "minecraft:orange_glazed_terracotta": "orange_glazed_terracotta",
    "minecraft:red_glazed_terracotta": "red_glazed_terracotta",
    "minecraft:green_glazed_terracotta": "green_glazed_terracotta",
    "minecraft:blue_glazed_terracotta": "blue_glazed_terracotta",
    "minecraft:cyan_glazed_terracotta": "cyan_glazed_terracotta",
    "minecraft:quartz_block": "quartz_block",
    "minecraft:quartz_pillar": "quartz_block 2",
    "minecraft:quartz_slab": "stone_slab 7",
    "minecraft:quartz_stairs": "quartz_stairs",
    "minecraft:smooth_quartz": "quartz_block",
    "minecraft:grass_block": "grass",
    "minecraft:dirt": "dirt",
    "minecraft:water": "water",
    "minecraft:oak_leaves": "leaves",
    "minecraft:birch_leaves": "leaves 2",
    "minecraft:spruce_leaves": "leaves 1",
    "minecraft:dark_prismarine": "prismarine 2",
    "minecraft:dark_prismarine_slab": "stone_slab",
    "minecraft:sea_lantern": "sea_lantern",
    "minecraft:glowstone": "glowstone",
    "minecraft:lantern": "glowstone",
    "minecraft:end_rod": "end_rod",
    "minecraft:sandstone": "sandstone",
    "minecraft:cut_sandstone": "sandstone 2",
    "minecraft:bookshelf": "bookshelf",
    "minecraft:air": "air",
    "minecraft:stone_brick_stairs": "stone_brick_stairs",
    "minecraft:smooth_sandstone_stairs": "sandstone_stairs",
    "minecraft:deepslate_tile_stairs": "stone_brick_stairs",
    "minecraft:stone_brick_slab": "stone_slab 5",
    "minecraft:stone_slab": "stone_slab",
    "minecraft:redstone_block": "redstone_block",
    "minecraft:red_nether_bricks": "red_nether_brick",
    "minecraft:nether_bricks": "nether_brick",
    "minecraft:magma_block": "magma",
    "minecraft:lava": "lava",
    "minecraft:quartz_stairs_north": "quartz_stairs 3",
    "minecraft:quartz_stairs_south": "quartz_stairs 2",
    "minecraft:quartz_stairs_east": "quartz_stairs 0",
    "minecraft:quartz_stairs_west": "quartz_stairs 1",
}


def _convert_block_id(modern_id: str) -> str:
    """Convert modern Minecraft block ID to Malmo/1.11.2 compatible ID."""
    if modern_id in BLOCK_ID_MAP:
        return BLOCK_ID_MAP[modern_id]
    short = modern_id.replace("minecraft:", "")
    if short in BLOCK_ID_MAP:
        return BLOCK_ID_MAP[short]
    return short


def _empty_mission_xml(time_limit_ms: int = 1200000) -> str:
    """Minimal mission XML: flat world, creative mode, chat commands enabled."""
    return f'''<?xml version="1.0" encoding="UTF-8" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <About><Summary>Rotterdam Builder</Summary></About>
    <ModSettings><MsPerTick>50</MsPerTick></ModSettings>
    <ServerSection>
        <ServerInitialConditions>
            <Time><StartTime>6000</StartTime><AllowPassageOfTime>false</AllowPassageOfTime></Time>
            <Weather>clear</Weather>
            <AllowSpawning>false</AllowSpawning>
        </ServerInitialConditions>
        <ServerHandlers>
            <FlatWorldGenerator generatorString="3;7,2x3,2;1;" forceReset="true"/>
            <ServerQuitFromTimeUp timeLimitMs="{time_limit_ms}"/>
            <ServerQuitWhenAnyAgentFinishes/>
        </ServerHandlers>
    </ServerSection>
    <AgentSection mode="Creative">
        <Name>Architect</Name>
        <AgentStart><Placement x="0" y="5" z="-50" yaw="0"/></AgentStart>
        <AgentHandlers>
            <ObservationFromFullStats/>
            <ContinuousMovementCommands turnSpeedDegs="180"/>
            <AbsoluteMovementCommands/>
            <ChatCommands/>
        </AgentHandlers>
    </AgentSection>
</Mission>'''


def run_in_malmo(
    buildings: list,
    origin_y: int = 4,
    spacing: int = 200,
    batch_size: int = 100,
):
    """Connect to Minecraft+Malmo and build structures using /setblock commands.

    Starts an empty flat world, then places blocks via chat commands in batches.

    Args:
        buildings: List of (name, World) tuples.
        origin_y: Ground Y level (4 for flat world).
        spacing: X spacing between buildings.
        batch_size: Blocks per batch (sleep between batches).
    """
    import time
    import sys
    # Add Malmo path if needed
    malmo_path = r"C:\Users\hiwa\Downloads\old\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\Malmo_Python3.7\Python_Examples"
    if malmo_path not in sys.path:
        sys.path.insert(0, malmo_path)
    try:
        from MalmoPython import AgentHost, MissionSpec, MissionRecordSpec
    except ImportError:
        from malmo.MalmoPython import AgentHost, MissionSpec, MissionRecordSpec

    agent_host = AgentHost()

    xml = _empty_mission_xml()
    mission = MissionSpec(xml, True)
    record = MissionRecordSpec()

    total_blocks = sum(w.block_count() for _, w in buildings)
    print(f"Starting mission ({total_blocks:,} total blocks)...")

    for attempt in range(10):
        try:
            agent_host.startMission(mission, record)
            print("Connected!")
            break
        except RuntimeError as e:
            print(f"  Attempt {attempt + 1}: {e}")
            time.sleep(3)
    else:
        raise RuntimeError("Failed to connect after 10 attempts")

    ws = agent_host.getWorldState()
    while not ws.has_mission_begun:
        time.sleep(0.1)
        ws = agent_host.getWorldState()
        if ws.errors:
            for error in ws.errors:
                print(f"Error: {error.text}")
            return

    print("Mission started! Placing blocks via /setblock...")
    time.sleep(2)  # Let world fully load

    current_x = 0
    placed = 0

    for name, world in buildings:
        blocks = world.blocks()
        min_y = min(pos[1] for pos in blocks) if blocks else 0
        y_offset = origin_y - min_y

        print(f"\n  Placing {name} ({len(blocks):,} blocks) at x={current_x}...")

        for (x, y, z), block in blocks.items():
            wx = x + current_x
            wy = y + y_offset
            wz = z
            malmo_block = _convert_block_id(block)
            parts = malmo_block.split(" ")
            block_name = parts[0]
            block_data = parts[1] if len(parts) > 1 else "0"

            try:
                agent_host.sendCommand(f"chat /setblock {wx} {wy} {wz} {block_name} {block_data}")
            except Exception:
                pass

            placed += 1

            if placed % batch_size == 0:
                time.sleep(0.05)
                ws = agent_host.getWorldState()
                if not ws.is_mission_running:
                    print("\n  Mission ended unexpectedly!")
                    return
                pct = placed / total_blocks * 100
                print(f"\r    {placed:,}/{total_blocks:,} ({pct:.0f}%)", end="", flush=True)

        # Spawn LOTS of different non-aggressive fish in water
        water_positions = [(x, y, z) for (x, y, z), b in blocks.items()
                           if b in ("minecraft:water",)]
        if water_positions:
            import random
            rng = random.Random(42)

            # All non-aggressive water/fish mobs across versions
            # 1.11.2 has: squid
            # 1.13+  has: cod, salmon, tropical_fish, pufferfish
            # We summon ALL types — unsupported ones silently fail
            fish_types = [
                "squid",
                "squid",
                "cod",
                "salmon",
                "tropical_fish",
                "pufferfish",
                "cod",
                "salmon",
                "tropical_fish",
                "squid",
            ]

            # Spawn a LOT — up to 500 fish
            fish_count = min(500, len(water_positions))
            fish_spots = rng.sample(water_positions, fish_count)

            print(f"\n  Spawning {fish_count} fish (squid, cod, salmon, tropical_fish, pufferfish)...")
            for i, (fx, fy, fz) in enumerate(fish_spots):
                wx = fx + current_x
                wy = fy + y_offset
                wz = fz
                ftype = fish_types[i % len(fish_types)]
                try:
                    agent_host.sendCommand(f"chat /summon {ftype} {wx} {wy} {wz}")
                except Exception:
                    pass
                if i % 30 == 0:
                    time.sleep(0.05)
            time.sleep(0.3)
            print(f"  {fish_count} fish spawned!")

        dims = world.dimensions()
        current_x += max(dims[0], dims[2]) + spacing
        print(f"\n  {name} done!")

    print(f"\nAll {total_blocks:,} blocks placed!")
    print("\n  Fly around to explore (double-tap SPACE)")
    print("  Buildings along X axis:")
    cx = 0
    for name, w in buildings:
        dims = w.dimensions()
        print(f"    x={cx:>6}  {name}")
        cx += max(dims[0], dims[2]) + spacing
    print("\n  Press Ctrl+C to end")

    try:
        while ws.is_mission_running:
            time.sleep(1)
            ws = agent_host.getWorldState()
    except KeyboardInterrupt:
        print("\nEnding mission...")

    print("Mission ended.")


def save_mission_xml(
    world: World,
    filename: str,
    building_name: str = "Rotterdam Building",
    origin_x: int = 0,
    origin_y: int = 4,
    origin_z: int = 0,
) -> str:
    """Save mission XML (for offline use)."""
    xml = _empty_mission_xml()
    out_path = f"{filename}.xml"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    return out_path
