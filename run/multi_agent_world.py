"""Multi-agent Malmo training world — 4 agents learning to be social.

Launches 4 Malmo clients and starts a shared mission where agents
can see each other, share resources, and learn to communicate.

Prerequisites:
    Launch 4 Malmo clients (each on a different port):
        cd C:\\Users\\hiwa\\Malmo_Python3.7\\Minecraft
        launchClient.bat --port 10000
        launchClient.bat --port 10001
        launchClient.bat --port 10002
        launchClient.bat --port 10003

Usage:
    conda activate malmo
    python run/multi_agent_world.py
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from MalmoPython import AgentHost, MissionSpec, MissionRecordSpec, ClientPool, ClientInfo
except ImportError:
    from malmo.MalmoPython import AgentHost, MissionSpec, MissionRecordSpec, ClientPool, ClientInfo


NUM_AGENTS = 4
BASE_PORT = 10000
WORLD_SIZE = 40     # 40x40 training arena
WALL_HEIGHT = 3
TICK_RATE = 50      # ms per tick


def build_mission_xml():
    """Build multi-agent mission XML with a small training world.

    World layout (40x40):
        - Flat grass arena with stone brick walls
        - 4 food zones (one per quadrant) with wheat/carrots
        - Central meeting area with glowstone lighting
        - Small lava hazard patches
        - Water feature in center
        - 4 spawn points (one per corner)
    """
    # Spawn positions — one per corner of the arena
    spawns = [
        (-15, 7, -15, 135),   # NW corner, facing SE
        (15, 7, -15, 225),    # NE corner, facing SW
        (-15, 7, 15, 45),     # SW corner, facing NE
        (15, 7, 15, 315),     # SE corner, facing NW
    ]

    # Drawing commands for the world
    drawings = []

    # Base platform — grass floor
    drawings.append(
        f'<DrawCuboid x1="-{WORLD_SIZE//2}" y1="4" z1="-{WORLD_SIZE//2}" '
        f'x2="{WORLD_SIZE//2}" y2="4" z2="{WORLD_SIZE//2}" type="grass"/>'
    )
    # Dirt below
    drawings.append(
        f'<DrawCuboid x1="-{WORLD_SIZE//2}" y1="1" z1="-{WORLD_SIZE//2}" '
        f'x2="{WORLD_SIZE//2}" y2="3" z2="{WORLD_SIZE//2}" type="dirt"/>'
    )

    # Stone brick walls around arena
    h = WORLD_SIZE // 2
    for y in range(5, 5 + WALL_HEIGHT):
        drawings.append(f'<DrawCuboid x1="-{h}" y1="{y}" z1="-{h}" x2="{h}" y2="{y}" z2="-{h}" type="stonebrick"/>')
        drawings.append(f'<DrawCuboid x1="-{h}" y1="{y}" z1="{h}" x2="{h}" y2="{y}" z2="{h}" type="stonebrick"/>')
        drawings.append(f'<DrawCuboid x1="-{h}" y1="{y}" z1="-{h}" x2="-{h}" y2="{y}" z2="{h}" type="stonebrick"/>')
        drawings.append(f'<DrawCuboid x1="{h}" y1="{y}" z1="-{h}" x2="{h}" y2="{y}" z2="{h}" type="stonebrick"/>')

    # Central meeting area — quartz platform with glowstone lighting
    drawings.append('<DrawCuboid x1="-3" y1="4" z1="-3" x2="3" y2="4" z2="3" type="quartz_block"/>')
    drawings.append('<DrawBlock x="0" y="4" z="0" type="glowstone"/>')
    drawings.append('<DrawBlock x="-2" y="4" z="-2" type="glowstone"/>')
    drawings.append('<DrawBlock x="2" y="4" z="-2" type="glowstone"/>')
    drawings.append('<DrawBlock x="-2" y="4" z="2" type="glowstone"/>')
    drawings.append('<DrawBlock x="2" y="4" z="2" type="glowstone"/>')

    # Small water pool in center
    drawings.append('<DrawCuboid x1="-1" y1="4" z1="-1" x2="1" y2="4" z2="1" type="water"/>')

    # Food zones — 4 quadrants, each with a small crop area
    food_zones = [
        (-12, -12),   # NW quadrant
        (10, -12),    # NE quadrant
        (-12, 10),    # SW quadrant
        (10, 10),     # SE quadrant
    ]
    for fx, fz in food_zones:
        for dx in range(3):
            for dz in range(3):
                drawings.append(f'<DrawBlock x="{fx+dx}" y="4" z="{fz+dz}" type="farmland"/>')
                drawings.append(f'<DrawBlock x="{fx+dx}" y="5" z="{fz+dz}" type="wheat"/>')

    # Lava hazards — small patches agents must learn to avoid
    lava_spots = [(-8, 0), (8, 0), (0, -8), (0, 8)]
    for lx, lz in lava_spots:
        drawings.append(f'<DrawBlock x="{lx}" y="4" z="{lz}" type="lava"/>')
        drawings.append(f'<DrawBlock x="{lx+1}" y="4" z="{lz}" type="lava"/>')

    # Torches for visibility
    for tx, tz in [(-10, -10), (10, -10), (-10, 10), (10, 10),
                    (-10, 0), (10, 0), (0, -10), (0, 10)]:
        drawings.append(f'<DrawBlock x="{tx}" y="5" z="{tz}" type="torch"/>')

    drawing_xml = "\n                ".join(drawings)

    # Build agent sections
    agent_sections = ""
    agent_names = ["Adam", "Eve", "Cain", "Abel"]
    colors = ["RED", "BLUE", "GREEN", "YELLOW"]

    for i in range(NUM_AGENTS):
        x, y, z, yaw = spawns[i]
        name = agent_names[i]

        agent_sections += f'''
    <AgentSection mode="Survival">
        <Name>{name}</Name>
        <AgentStart>
            <Placement x="{x}" y="{y}" z="{z}" yaw="{yaw}"/>
        </AgentStart>
        <AgentHandlers>
            <ObservationFromFullStats/>
            <ObservationFromGrid>
                <Grid name="floor3x3">
                    <min x="-1" y="-1" z="-1"/>
                    <max x="1" y="0" z="1"/>
                </Grid>
                <Grid name="view5x5">
                    <min x="-2" y="-1" z="-2"/>
                    <max x="2" y="1" z="2"/>
                </Grid>
                <Grid name="view7x7">
                    <min x="-3" y="-1" z="-3"/>
                    <max x="3" y="1" z="3"/>
                </Grid>
            </ObservationFromGrid>
            <ObservationFromNearbyEntities>
                <Range name="nearby_entities" xrange="20" yrange="5" zrange="20"/>
            </ObservationFromNearbyEntities>
            <ObservationFromChat/>
            <ContinuousMovementCommands turnSpeedDegs="180"/>
            <ChatCommands/>
        </AgentHandlers>
    </AgentSection>'''

    xml = f'''<?xml version="1.0" encoding="UTF-8" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

    <About>
        <Summary>Multi-Agent Social Training World</Summary>
    </About>

    <ModSettings>
        <MsPerTick>{TICK_RATE}</MsPerTick>
    </ModSettings>

    <ServerSection>
        <ServerInitialConditions>
            <Time>
                <StartTime>6000</StartTime>
                <AllowPassageOfTime>false</AllowPassageOfTime>
            </Time>
            <Weather>clear</Weather>
            <AllowSpawning>false</AllowSpawning>
        </ServerInitialConditions>
        <ServerHandlers>
            <FlatWorldGenerator generatorString="3;7,2x3,2;1;" forceReset="true"/>
            <DrawingDecorator>
                {drawing_xml}
            </DrawingDecorator>
            <ServerQuitWhenAnyAgentFinishes/>
        </ServerHandlers>
    </ServerSection>
    {agent_sections}
</Mission>'''

    return xml


def launch_agents():
    """Create 4 agent hosts and connect them to the mission."""
    # Create agent hosts
    agents = []
    for i in range(NUM_AGENTS):
        host = AgentHost()
        agents.append(host)

    # Client pool — each agent connects to its own Minecraft client
    pool = ClientPool()
    for i in range(NUM_AGENTS):
        pool.add(ClientInfo('127.0.0.1', BASE_PORT + i))

    # Build mission
    xml = build_mission_xml()
    mission = MissionSpec(xml, True)
    record = MissionRecordSpec()

    print(f"Starting multi-agent mission with {NUM_AGENTS} agents...")
    print(f"  Ports: {[BASE_PORT + i for i in range(NUM_AGENTS)]}")

    # Start mission for each agent
    # Agent 0 is the "server" — starts first with role 0
    # Agents 1-3 join with role 1, 2, 3
    for i, host in enumerate(agents):
        for attempt in range(10):
            try:
                host.startMission(mission, pool, record, i, "social_training")
                print(f"  Agent {i} ({['Adam', 'Eve', 'Cain', 'Abel'][i]}) connected!")
                break
            except RuntimeError as e:
                print(f"    Attempt {attempt + 1}: {e}")
                time.sleep(3)
        else:
            print(f"  Agent {i} FAILED to connect after 10 attempts!")
            return None
        # Small delay between agent connections
        if i < NUM_AGENTS - 1:
            time.sleep(2)

    # Wait for all agents to start
    print("\nWaiting for mission to begin...")
    world_states = [None] * NUM_AGENTS
    all_started = False
    while not all_started:
        all_started = True
        for i, host in enumerate(agents):
            ws = host.getWorldState()
            world_states[i] = ws
            if not ws.has_mission_begun:
                all_started = False
        time.sleep(0.1)

    print("All 4 agents are in the world!")
    return agents, world_states


def training_loop(agents, world_states):
    """Main training loop — agents observe, decide, act."""
    import json

    agent_names = ["Adam", "Eve", "Cain", "Abel"]
    tick = 0

    print("\n" + "=" * 60)
    print("  TRAINING STARTED")
    print("  4 agents in a 40x40 arena")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        while True:
            tick += 1
            any_running = False

            for i, host in enumerate(agents):
                ws = host.getWorldState()
                world_states[i] = ws

                if not ws.is_mission_running:
                    continue
                any_running = True

                # Get observations
                if ws.number_of_observations_since_last_state > 0:
                    obs = json.loads(ws.observations[-1].text)

                    # Print status every 100 ticks
                    if tick % 100 == 0 and i == 0:
                        life = obs.get("Life", "?")
                        food = obs.get("Food", "?")
                        x = obs.get("XPos", 0)
                        z = obs.get("ZPos", 0)
                        nearby = obs.get("nearby_entities", [])
                        print(f"  Tick {tick:>6} | {agent_names[i]:>5} "
                              f"pos=({x:.0f},{z:.0f}) "
                              f"life={life} food={food} "
                              f"nearby={len(nearby)} entities")

                    # === AGENT BEHAVIOR (placeholder for brain modules) ===
                    # For now: simple random exploration so agents move around
                    # This will be replaced by the brain architecture

                    import random
                    action = random.choice([
                        ("move", 0.5),      # walk forward
                        ("move", 0),        # stop
                        ("turn", 0.3),      # turn right
                        ("turn", -0.3),     # turn left
                        ("move", -0.2),     # back up
                    ])

                    try:
                        host.sendCommand(f"{action[0]} {action[1]}")
                    except Exception:
                        pass

                    # Occasionally vocalize (placeholder for communication)
                    if tick % 50 == i * 12:
                        letter = random.choice(["A", "B", "C", "D", "E"])
                        try:
                            host.sendCommand(f"chat {agent_names[i]}: {letter}")
                        except Exception:
                            pass

            if not any_running:
                print("\nAll agents' missions ended.")
                break

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nTraining stopped by user.")

    print(f"\nTotal ticks: {tick}")


def main():
    print("=" * 60)
    print("  MULTI-AGENT SOCIAL TRAINING WORLD")
    print("  4 agents learning together from scratch")
    print("=" * 60)
    print()
    print("Make sure 4 Malmo clients are running:")
    print(f"  Ports: {[BASE_PORT + i for i in range(NUM_AGENTS)]}")
    print()

    result = launch_agents()
    if result is None:
        print("\nFailed to launch agents. Make sure all 4 Malmo clients are running.")
        print("To launch them:")
        print("  cd C:\\Users\\hiwa\\Malmo_Python3.7\\Minecraft")
        print("  launchClient.bat --port 10000")
        print("  launchClient.bat --port 10001")
        print("  launchClient.bat --port 10002")
        print("  launchClient.bat --port 10003")
        return

    agents, world_states = result
    training_loop(agents, world_states)


if __name__ == "__main__":
    main()
