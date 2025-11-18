# test_01_start_mission.py
from MalmoPython import AgentHost, MissionSpec, MissionRecordSpec
import time

agent = AgentHost()

mission_xml = """<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://ProjectMalmo.microsoft.com
         https://raw.githubusercontent.com/Microsoft/malmo/master/Schemas/Mission.xsd">

  <About>
    <Summary>Test mission start</Summary>
  </About>

  <ServerSection>
    <ServerInitialConditions>
      <Time>
        <StartTime>1000</StartTime>
        <AllowPassageOfTime>true</AllowPassageOfTime>
      </Time>
    </ServerInitialConditions>

    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,2;1;"/>
    </ServerHandlers>
  </ServerSection>

  <AgentSection mode="Survival">
    <Name>TestAgent</Name>
    <AgentStart/>
    <AgentHandlers>
      <ObservationFromFullStats/>
      <ContinuousMovementCommands/>
    </AgentHandlers>
  </AgentSection>
</Mission>
"""

mission = MissionSpec(mission_xml, True)
record = MissionRecordSpec()

print("Starting mission...")
agent.startMission(mission, record)

# Wait for mission to start
world_state = agent.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="", flush=True)
    time.sleep(0.1)
    world_state = agent.getWorldState()

print("\nMission started OK.")

agent.sendCommand("move 1")
time.sleep(2)
agent.sendCommand("move 0")

print("Movement test complete.")