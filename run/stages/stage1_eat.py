"""Stage 1: Food in reach — learn to eat.

Brain region: Brainstem
Setup: Open 16x16 grassy yard with fence walls. Daytime.
       Food (cake) scattered near each agent. Respawn at center.
New capability: Sensation + motor output
Success: Agent eats when food is adjacent.

4 agents, 7x7 vision, 5x speed, single respawn point.
"""

NUM_AGENTS = 4
AGENT_NAMES = ["Adam", "Eve", "Cain", "Abel"]
ARENA_SIZE = 16
RESPAWN = (0, 5, 0)  # Center of the yard
MS_PER_TICK_DEFAULT = 50  # normal speed, overridden by --speed flag


def mission_xml(time_limit_ms: int = 0, headless: bool = False, ms_per_tick: int = None) -> str:
    """Build Stage 1 mission XML.

    Open grassy yard in daytime. Fence walls, flowers, a tree, and
    cake blocks scattered everywhere. Feels like a small garden.
    """
    if ms_per_tick is None:
        ms_per_tick = MS_PER_TICK_DEFAULT
    h = ARENA_SIZE // 2  # 8

    # Agents spawn in 4 quadrants, facing center
    spawns = [
        (-4, 5, -4, 135),   # NW → face SE
        (4, 5, -4, 225),    # NE → face SW
        (-4, 5, 4, 45),     # SW → face NE
        (4, 5, 4, 315),     # SE → face NW
    ]

    d = []  # drawing commands

    # ── Ground ──
    # Dirt base
    d.append(f'<DrawCuboid x1="-{h}" y1="1" z1="-{h}" x2="{h}" y2="3" z2="{h}" type="dirt"/>')
    # Grass top
    d.append(f'<DrawCuboid x1="-{h}" y1="4" z1="-{h}" x2="{h}" y2="4" z2="{h}" type="grass"/>')

    # ── Fence walls (1 block high on grass) ──
    for x in range(-h, h + 1):
        d.append(f'<DrawBlock x="{x}" y="5" z="-{h}" type="fence"/>')
        d.append(f'<DrawBlock x="{x}" y="5" z="{h}" type="fence"/>')
    for z in range(-h + 1, h):
        d.append(f'<DrawBlock x="-{h}" y="5" z="{z}" type="fence"/>')
        d.append(f'<DrawBlock x="{h}" y="5" z="{z}" type="fence"/>')

    # ── Small path (gravel cross in center) ──
    for i in range(-2, 3):
        d.append(f'<DrawBlock x="{i}" y="4" z="0" type="gravel"/>')
        d.append(f'<DrawBlock x="0" y="4" z="{i}" type="gravel"/>')

    # ── Respawn marker (center) ──
    d.append('<DrawBlock x="0" y="4" z="0" type="quartz_block"/>')

    # ── Small tree ──
    # Trunk
    for y in range(5, 8):
        d.append(f'<DrawBlock x="-6" y="{y}" z="-6" type="log"/>')
    # Leaves
    for dx in range(-1, 2):
        for dz in range(-1, 2):
            d.append(f'<DrawBlock x="{-6+dx}" y="8" z="{-6+dz}" type="leaves"/>')
            if abs(dx) + abs(dz) <= 1:
                d.append(f'<DrawBlock x="{-6+dx}" y="9" z="{-6+dz}" type="leaves"/>')

    # ── Flowers (decoration, not food) ──
    flowers = [
        (3, 5, -6, "red_flower"), (-5, 5, 3, "yellow_flower"),
        (6, 5, 5, "red_flower"), (-3, 5, -5, "yellow_flower"),
        (5, 5, -3, "red_flower"), (-6, 5, 6, "yellow_flower"),
    ]
    for fx, fy, fz, ft in flowers:
        d.append(f'<DrawBlock x="{fx}" y="{fy}" z="{fz}" type="{ft}"/>')

    # ── Food: cake blocks scattered near each spawn ──
    # Lots of food, close to agents — Stage 1 should be easy to find
    food = [
        # Near Adam (NW)
        (-3, 5, -3), (-5, 5, -3), (-3, 5, -5), (-4, 5, -2),
        # Near Eve (NE)
        (3, 5, -3), (5, 5, -3), (3, 5, -5), (4, 5, -2),
        # Near Cain (SW)
        (-3, 5, 3), (-5, 5, 3), (-3, 5, 5), (-4, 5, 2),
        # Near Abel (SE)
        (3, 5, 3), (5, 5, 3), (3, 5, 5), (4, 5, 2),
        # Center area
        (-1, 5, -1), (1, 5, -1), (-1, 5, 1), (1, 5, 1), (0, 5, 0),
    ]
    for fx, fy, fz in food:
        d.append(f'<DrawBlock x="{fx}" y="{fy}" z="{fz}" type="cake"/>')

    drawing_xml = "\n                ".join(d)

    # Server quit
    quit_xml = ""
    if time_limit_ms > 0:
        quit_xml = f'<ServerQuitFromTimeUp timeLimitMs="{time_limit_ms}"/>'

    # Agent sections
    agent_sections = ""
    for i in range(NUM_AGENTS):
        x, y, z, yaw = spawns[i]
        name = AGENT_NAMES[i]

        agent_sections += f'''
    <AgentSection mode="Survival">
        <Name>{name}</Name>
        <AgentStart>
            <Placement x="{x}" y="{y}" z="{z}" yaw="{yaw}"/>
        </AgentStart>
        <AgentHandlers>
            <ObservationFromFullStats/>
            <ObservationFromHotBar/>
            <ObservationFromFullInventory/>
            <ObservationFromGrid>
                <Grid name="view7x7">
                    <min x="-3" y="-1" z="-3"/>
                    <max x="3" y="1" z="3"/>
                </Grid>
            </ObservationFromGrid>
            <ObservationFromNearbyEntities>
                <Range name="nearby" xrange="{ARENA_SIZE}" yrange="3" zrange="{ARENA_SIZE}"/>
            </ObservationFromNearbyEntities>
            <ContinuousMovementCommands turnSpeedDegs="180"/>
            <InventoryCommands/>
            <AbsoluteMovementCommands/>
            <ChatCommands/>
        </AgentHandlers>
    </AgentSection>'''

    xml = f'''<?xml version="1.0" encoding="UTF-8" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <About>
        <Summary>Stage 1: Food in reach - learn to eat</Summary>
    </About>
    <ModSettings>
        <MsPerTick>{ms_per_tick}</MsPerTick>
    </ModSettings>
    <ServerSection>
        <ServerInitialConditions>
            <Time>
                <StartTime>1000</StartTime>
                <AllowPassageOfTime>true</AllowPassageOfTime>
            </Time>
            <Weather>clear</Weather>
            <AllowSpawning>false</AllowSpawning>
        </ServerInitialConditions>
        <ServerHandlers>
            <FlatWorldGenerator generatorString="3;7,2x3,2;1;" forceReset="true"/>
            <DrawingDecorator>
                {drawing_xml}
            </DrawingDecorator>
            {quit_xml}
        </ServerHandlers>
    </ServerSection>
    {agent_sections}
</Mission>'''

    return xml
