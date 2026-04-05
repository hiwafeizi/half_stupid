"""Stage 1 Training: Food in reach — learn to eat.

4 agents in a 16x16 grassy yard with food nearby.
Reward: ONLY survival (+1 alive, -100 death). Nothing else. Ever.
Brain: 3-layer NN (115 inputs → 128 → 64 → 23 actions).

Usage:
    python run/start.py --episodes 10 --speed 5
    python run/start.py --episodes 10 --skip-launch --speed 20
"""

import sys
import os
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from MalmoPython import AgentHost, MissionSpec, MissionRecordSpec, ClientPool, ClientInfo
except ImportError:
    from malmo.MalmoPython import AgentHost, MissionSpec, MissionRecordSpec, ClientPool, ClientInfo

from run.stages.stage1_eat import mission_xml, NUM_AGENTS, AGENT_NAMES, RESPAWN
from brain.brainstem import Brainstem, ACTIONS, NUM_ACTIONS


# ═══════════════════════════════════════════════════════════════
#  CONFIG — change these freely
# ═══════════════════════════════════════════════════════════════
BASE_PORT = 10000
SAVE_DIR = "run/checkpoints/stage1"
EPISODE_TIME_MS = 900000      # 15 minutes per episode
GAME_SPEED = 20                # 20x. Python can keep up now (no sleep in loop).
DEFAULT_EPISODES = 50         # default number of episodes

# ═══════════════════════════════════════════════════════════════
#  REWARD — DO NOT CHANGE. Only survival. See feedback_reward.md
# ═══════════════════════════════════════════════════════════════
REWARD_ALIVE = 1.0            # +1 per tick alive
REWARD_DEATH = -100.0         # -100 on death


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: Learn to eat")
    p.add_argument("--headless", action="store_true", help="Run without UI")
    p.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Number of episodes")
    p.add_argument("--load", type=str, default=None, help="Load weights from file")
    p.add_argument("--load-dir", type=str, default=None, help="Load all agents from checkpoint dir")
    p.add_argument("--speed", type=int, default=GAME_SPEED,
                   help=f"Game speed multiplier (default: {GAME_SPEED}x)")
    p.add_argument("--ports", type=str, default="10000,10001,10002,10003",
                   help="Comma-separated ports for 4 Malmo clients")
    return p.parse_args()


def connect_agents(ports: list, xml: str) -> list:
    """Connect 4 agents to their Malmo clients and start the mission."""
    agents = []
    pool = ClientPool()
    for port in ports:
        pool.add(ClientInfo('127.0.0.1', port))

    mission = MissionSpec(xml, True)

    for i in range(NUM_AGENTS):
        host = AgentHost()
        agents.append(host)
        record = MissionRecordSpec()

        for attempt in range(10):
            try:
                host.startMission(mission, pool, record, i, "stage1")
                print(f"  {AGENT_NAMES[i]} connected (port {ports[i]})")
                break
            except RuntimeError as e:
                if attempt == 9:
                    print(f"  {AGENT_NAMES[i]} FAILED: {e}")
                    return None
                time.sleep(3)

        if i < NUM_AGENTS - 1:
            time.sleep(1)

    # Wait for all to start
    print("  Waiting for mission...", end="", flush=True)
    while True:
        all_started = all(
            agents[i].getWorldState().has_mission_begun
            for i in range(NUM_AGENTS)
        )
        if all_started:
            break
        time.sleep(0.1)
        print(".", end="", flush=True)
    print(" Started!")

    return agents


def respawn_agent(host, name: str):
    """Teleport agent back to respawn point."""
    rx, ry, rz = RESPAWN
    try:
        host.sendCommand(f"tp {rx} {ry} {rz}")
    except Exception:
        pass


HUNGER_DRAIN_EVERY = 30    # Drain 1 food level every 30 ticks
                           # 20 food / 1 per 30 ticks = 600 ticks to starve = ~6 sec
                           # When food hits 0, health starts dropping


def run_episode(agents: list, brains: list, episode: int) -> dict:
    """Run one training episode. Returns stats."""
    tick = 0
    alive_ticks = [0] * NUM_AGENTS
    deaths = [0] * NUM_AGENTS
    food_eaten = [0] * NUM_AGENTS
    prev_food = [20.0] * NUM_AGENTS

    while True:
        tick += 1
        any_running = False

        for i in range(NUM_AGENTS):
            ws = agents[i].getWorldState()
            if not ws.is_mission_running:
                continue
            any_running = True

            if ws.number_of_observations_since_last_state == 0:
                continue

            obs = json.loads(ws.observations[-1].text)

            # Check if agent died (health <= 0)
            health = obs.get("Life", 0)
            food_level = obs.get("Food", 0)

            if health <= 0:
                deaths[i] += 1
                respawn_agent(agents[i], AGENT_NAMES[i])
                brains[i].record_reward(REWARD_DEATH)
                continue

            alive_ticks[i] += 1

            # Apply hunger effect — drains food bar, then health on empty stomach
            # Reapply every N ticks to keep it going
            if tick % HUNGER_DRAIN_EVERY == 0:
                try:
                    agents[i].sendCommand("chat /effect @p 17 5 127 true")  # hunger, 5sec, max level, silent
                except Exception:
                    pass

            # Track food for stats only — reward is ONLY survival
            if food_level > prev_food[i]:
                food_eaten[i] += 1
            prev_food[i] = food_level

            reward = REWARD_ALIVE  # +1 alive. Nothing else. Ever.

            # Agent chooses action (observe → encode → forward → sample)
            grid = obs.get("view7x7", [])
            action_idx = brains[i].choose_action(obs, grid)

            # Execute action (like pressing a keyboard key)
            cmd, val = ACTIONS[action_idx]
            try:
                agents[i].sendCommand(f"{cmd} {val}")
            except Exception:
                pass

            brains[i].record_reward(reward)

        if not any_running:
            break

    # End of episode — update all brains
    avg_rewards = []
    for i in range(NUM_AGENTS):
        avg_r = brains[i].update()
        avg_rewards.append(avg_r)

    return {
        "episode": episode,
        "ticks": tick,
        "alive_ticks": alive_ticks,
        "deaths": deaths,
        "food_eaten": food_eaten,
        "avg_rewards": avg_rewards,
    }


def main():
    args = parse_args()
    ports = [int(p) for p in args.ports.split(",")]

    ep_min = EPISODE_TIME_MS / 60000
    ms_per_tick = max(1, 50 // args.speed)
    ticks_per_sec = 1000 // ms_per_tick
    print("=" * 60)
    print("  STAGE 1: Food in reach — learn to eat")
    print(f"  4 agents | 7x7 vision | 3-layer NN | {args.speed}x speed")
    print(f"  Episode: {ep_min:.0f} min | {ms_per_tick}ms/tick | {ticks_per_sec} ticks/sec")
    print(f"  Reward: +1 alive, -100 death (LOCKED)")
    print("=" * 60)
    print()

    # Create brains — each agent gets its own
    brains = []
    for i in range(NUM_AGENTS):
        name = AGENT_NAMES[i]
        brain = Brainstem(name=name, learning_rate=0.001)

        # Resume from checkpoint if available
        load_dir = args.load_dir or args.load
        if load_dir:
            agent_dir = os.path.join(load_dir, name)
            if os.path.exists(agent_dir):
                brain.load(agent_dir)
                print(f"  Loaded {name} from {agent_dir} (ep={brain.episodes_trained})")

        brains.append(brain)

    # Training loop
    all_stats = []
    for ep in range(1, args.episodes + 1):
        print(f"\n--- Episode {ep}/{args.episodes} ---")

        # Generate mission XML
        ms_per_tick = max(1, 50 // args.speed)  # normal=50ms, speed 5=10ms, speed 50=1ms
        xml = mission_xml(time_limit_ms=EPISODE_TIME_MS, ms_per_tick=ms_per_tick)

        # Connect
        agents = connect_agents(ports, xml)
        if agents is None:
            print("Failed to connect. Make sure 4 Malmo clients are running.")
            print("Run: run\\launch_4_clients.bat")
            return

        # Setup: hard difficulty, fill hotbar with fruits
        # NO /effect for hunger — we drain food from Python so it scales with speed
        time.sleep(0.5)
        try:
            agents[0].sendCommand("chat /difficulty hard")
        except Exception:
            pass
        time.sleep(0.2)

        # Fill all 9 hotbar slots with different foods
        foods = [
            "apple 64",
            "bread 64",
            "cooked_porkchop 64",
            "cooked_beef 64",
            "cooked_chicken 64",
            "cooked_fish 64",
            "melon 64",
            "carrot 64",
            "baked_potato 64",
        ]
        for i in range(NUM_AGENTS):
            try:
                agents[i].sendCommand("chat /clear @p")
                time.sleep(0.05)
                for food in foods:
                    agents[i].sendCommand(f"chat /give @p {food}")
                    time.sleep(0.02)
            except Exception:
                pass

        # Run episode
        stats = run_episode(agents, brains, ep)
        all_stats.append(stats)

        # Print stats
        print(f"  Ticks: {stats['ticks']}")
        for i in range(NUM_AGENTS):
            print(f"  {AGENT_NAMES[i]:>5}: alive={stats['alive_ticks'][i]:>5} "
                  f"deaths={stats['deaths'][i]} "
                  f"food={stats['food_eaten'][i]} "
                  f"avg_r={stats['avg_rewards'][i]:.3f}")

        # Save after every episode so we never lose progress
        for i in range(NUM_AGENTS):
            agent_dir = os.path.join(SAVE_DIR, AGENT_NAMES[i])
            brains[i].save(agent_dir)
        print(f"  Saved to {SAVE_DIR}/")

        # Wait for Malmo to reset between episodes
        time.sleep(3)

    # Final save
    for i in range(NUM_AGENTS):
        agent_dir = os.path.join(SAVE_DIR, AGENT_NAMES[i])
        brains[i].save(agent_dir)

    # Summary
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    if all_stats:
        last = all_stats[-1]
        first = all_stats[0]
        print(f"  Episodes: {len(all_stats)}")
        print(f"  First avg reward: {sum(first['avg_rewards'])/4:.3f}")
        print(f"  Last  avg reward: {sum(last['avg_rewards'])/4:.3f}")
        print(f"  Weights saved to: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
