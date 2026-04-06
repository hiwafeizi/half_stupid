"""Stage 1 Training: Food in reach — learn to eat.

World: 16x16 grassy yard, daytime, fence walls, cake blocks nearby.
Agents: 4 (Adam, Eve, Cain, Abel), one life per episode.
Brain: 133 inputs -> 128 -> 64 -> 23 actions (REINFORCE).
Inputs: 98 vision + 7 body + 9 hotbar IDs + 9 counts + 2 held item + 8 action flags.
Reward: ONLY +1 alive, -10000 death. Nothing else. Ever.
Hunger: /effect hunger drains food bar, starvation kills on hard difficulty.
Masking: input and action levels configurable (1-5 each).
Saves: per-agent weights, vocab, full episode history. Auto-loads on restart.
Debug: live_<name>.json updated every ~1 second with full agent state.

Usage:
    python run/start.py --skip-launch --episodes 10
    python run/start.py --skip-launch --input-level 3 --action-level 2
"""

import sys
import os
import time
import json
import argparse
from datetime import datetime

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
EPISODE_TIME_MS = 3000000      # 50 minutes real time per episode
GAME_SPEED = 40                # How fast the Minecraft WORLD runs vs real time.
                               # 1 = real time. 20 = world runs 20x faster.
                               # Hunger drains 20x faster, day passes 20x faster.
                               # Malmo MsPerTick = 50 / GAME_SPEED (lower = faster world).
                               # The agent's brain speed is NOT affected by this.

MS_PER_TICK = 50               # Agent think rate in GAME time (ms).
                               # 50 = 20 decisions per game second.
                               # At GAME_SPEED=20, real sleep = 50/20 = 2.5ms = 400 decisions/real sec.
                               # GAME_SPEED only affects real time, not game logic.
DEFAULT_EPISODES = 50          # default number of episodes
LEARNING_RATE = 0.09           # start high for simple tasks, lower for complex stages

# ─── INPUT MASK LEVEL ─────────────────────────────────────────
# See brain/brainstem.py MASK_LEVELS for full details
# Level 1: health + food + eating flag (3 active)
# Level 2: + held item ID + held item count (5 active)
# Level 3: + 9 hotbar IDs + 9 hotbar counts (23 active)
# Level 4: + 98 vision + 7 movement action flags (128 active)
# Level 5: + x,y,z + yaw + pitch (133 active = all)
INPUT_MASK_LEVEL = 3

# ─── OUTPUT (ACTION) MASK LEVEL ──────────────────────────────
# See brain/brainstem.py ACTION_MASK_LEVELS for full details
# Level 1: eat + stand still (2 active)
# Level 2: + hotbar 1-3 (5 active)
# Level 3: + hotbar 4-9 (11 active)
# Level 4: + walk fwd/back + turn (15 active)
# Level 5: + strafe, jump, crouch, attack, throw (21 active)
# Level 6: + look up/down (23 active = all)
ACTION_MASK_LEVEL = 3

# ═══════════════════════════════════════════════════════════════
#  REWARD — DO NOT CHANGE. Only survival. See feedback_reward.md
# ═══════════════════════════════════════════════════════════════
REWARD_ALIVE = 1.0            # +1 per tick alive
REWARD_DEATH = -1000.0       # -10,000 on death


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: Learn to eat")
    p.add_argument("--headless", action="store_true", help="Run without UI")
    p.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Number of episodes")
    p.add_argument("--load", type=str, default=None, help="Load weights from file")
    p.add_argument("--load-dir", type=str, default=None, help="Load all agents from checkpoint dir")
    p.add_argument("--ms-per-tick", type=int, default=MS_PER_TICK,
                   help=f"Milliseconds per game tick (default: {MS_PER_TICK})")
    p.add_argument("--input-level", type=int, default=INPUT_MASK_LEVEL,
                   help=f"Input mask level 1-5 (default: {INPUT_MASK_LEVEL})")
    p.add_argument("--action-level", type=int, default=ACTION_MASK_LEVEL,
                   help=f"Action mask level 1-5 (default: {ACTION_MASK_LEVEL})")
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
    """Revive a dead agent: heal, feed, teleport, restock."""
    rx, ry, rz = RESPAWN
    try:
        # Switch to creative to unkill, then back to survival
        host.sendCommand(f"chat /gamemode 1 {name}")
        host.sendCommand(f"chat /tp {name} {rx} {ry} {rz}")
        host.sendCommand(f"chat /effect {name} 6 1 255 true")   # instant health max, silent
        host.sendCommand(f"chat /effect {name} 23 1 255 true")  # saturation max, silent
        host.sendCommand(f"chat /gamemode 0 {name}")
        # Restock food
        host.sendCommand(f"chat /give {name} apple 10")
        host.sendCommand(f"chat /give {name} bread 10")
        host.sendCommand(f"chat /give {name} cooked_beef 10")
    except Exception:
        pass


HUNGER_DRAIN_EVERY = 160    # Reapply hunger effect every 160 ticks
                           # Uses /effect hunger (ID 17) at max amplifier
                           # Drains food bar, then starvation kills on hard difficulty


def run_episode(agents: list, brains: list, episode: int) -> dict:
    """Run one training episode. Returns stats."""
    tick = 0
    alive_ticks = [0] * NUM_AGENTS
    dead = [False] * NUM_AGENTS
    food_eaten = [0] * NUM_AGENTS
    prev_food = [20.0] * NUM_AGENTS
    live_logs = {name: [] for name in AGENT_NAMES}  # accumulate every second
    # Track active continuous commands per agent
    active = [{"use": 0, "move": 0, "strafe": 0, "turn": 0,
               "pitch": 0, "jump": 0, "crouch": 0, "attack": 0}
              for _ in range(NUM_AGENTS)]

    while True:
        tick += 1
        any_running = False

        for i in range(NUM_AGENTS):
            # Already dead — skip for rest of episode
            if dead[i]:
                continue

            ws = agents[i].getWorldState()
            if not ws.is_mission_running:
                continue
            any_running = True

            if ws.number_of_observations_since_last_state == 0:
                continue

            obs = json.loads(ws.observations[-1].text)

            health = obs.get("Life", 0)
            food_level = obs.get("Food", 0)

            if health <= 0:
                dead[i] = True
                brains[i].record_reward(REWARD_DEATH)
                print(f"    {AGENT_NAMES[i]} DIED at tick {tick} (alive {alive_ticks[i]} ticks)")
                continue

            alive_ticks[i] += 1


            # Apply hunger effect — drains food bar, then health on empty stomach
            # Reapply every N ticks to keep it going
            if tick % HUNGER_DRAIN_EVERY == 0:
                try:
                    agents[i].sendCommand(f"chat /effect {AGENT_NAMES[i]} 17 5 127 true")  # hunger, 5sec, max level, silent
                except Exception:
                    pass

            # Track food for stats only — reward is ONLY survival
            if food_level > prev_food[i]:
                food_eaten[i] += 1
            prev_food[i] = food_level

            reward = REWARD_ALIVE  # +1 alive. Nothing else. Ever.

            # Agent chooses action (observe → encode → forward → sample)
            grid = obs.get("view7x7", [])
            action_idx = brains[i].choose_action(obs, grid, active[i])

            # Save live debug snapshot every ~1 second
            # Shows EXACTLY what the AI sees — raw numbers, same layout
            if tick % 500 == i:
                brain = brains[i]
                raw_input = brain._ep_inputs[-1].tolist() if brain._ep_inputs else []
                probs = brain._ep_probs[-1].tolist() if brain._ep_probs else []
                mask = brain.mask.tolist()
                action_mask = brain.action_mask.tolist()

                ms_tick = MS_PER_TICK
                snapshot = {
                    "agent": AGENT_NAMES[i],
                    "episode": episode,
                    "tick": tick,
                    "alive_ticks": alive_ticks[i],
                    "alive_seconds": round(alive_ticks[i] * ms_tick / 1000.0, 1),
                    "episode_seconds": round(tick * ms_tick / 1000.0, 1),

                    # === EXACTLY what the AI sees (133 floats) ===
                    "ai_input": {
                        "vision_98": [round(v, 1) for v in raw_input[0:98]] if len(raw_input) >= 98 else [],
                        "health": raw_input[98] if len(raw_input) > 98 else 0,
                        "food": raw_input[99] if len(raw_input) > 99 else 0,
                        "x": raw_input[100] if len(raw_input) > 100 else 0,
                        "y": raw_input[101] if len(raw_input) > 101 else 0,
                        "z": raw_input[102] if len(raw_input) > 102 else 0,
                        "yaw": raw_input[103] if len(raw_input) > 103 else 0,
                        "pitch": raw_input[104] if len(raw_input) > 104 else 0,
                        "hotbar_ids": [round(v, 1) for v in raw_input[105:114]] if len(raw_input) > 113 else [],
                        "hotbar_counts": [round(v, 1) for v in raw_input[114:123]] if len(raw_input) > 122 else [],
                        "held_item_id": raw_input[123] if len(raw_input) > 123 else 0,
                        "held_item_count": raw_input[124] if len(raw_input) > 124 else 0,
                        "active_eating": raw_input[125] if len(raw_input) > 125 else 0,
                        "active_moving": raw_input[126] if len(raw_input) > 126 else 0,
                        "active_strafing": raw_input[127] if len(raw_input) > 127 else 0,
                        "active_turning": raw_input[128] if len(raw_input) > 128 else 0,
                        "active_pitching": raw_input[129] if len(raw_input) > 129 else 0,
                        "active_jumping": raw_input[130] if len(raw_input) > 130 else 0,
                        "active_crouching": raw_input[131] if len(raw_input) > 131 else 0,
                        "active_attacking": raw_input[132] if len(raw_input) > 132 else 0,
                    },

                    # === EXACTLY what the AI sees after masking ===
                    "ai_input_masked": [round(v, 1) for v in (brain._ep_inputs[-1] * brain.mask).tolist()] if brain._ep_inputs else [],

                    # === AI's output (23 action probabilities) ===
                    "ai_output": {
                        "action_probs": [round(p, 8) for p in probs],
                        "chosen_action": action_idx,
                        "chosen_command": f"{ACTIONS[action_idx][0]} {ACTIONS[action_idx][1]}",
                    },

                    # === Masks (what's enabled) ===
                    "input_mask": mask,
                    "action_mask": action_mask,
                    "input_level": brain.input_level,
                    "action_level": brain.action_level,

                    # === Human-readable dictionary (separate from AI data) ===
                    "dictionary": brain.block_vocab,
                }
                live_logs[AGENT_NAMES[i]].append(snapshot)
                debug_path = os.path.join(SAVE_DIR, f"live_{AGENT_NAMES[i]}.json")
                os.makedirs(SAVE_DIR, exist_ok=True)
                with open(debug_path, "w") as f:
                    json.dump(live_logs[AGENT_NAMES[i]], f, indent=2)

            # Execute action and track active continuous commands
            cmd, val = ACTIONS[action_idx]
            try:
                if cmd.startswith("hotbar.") or cmd == "discardCurrentItem":
                    # Discrete: press+release, stop eating first
                    if active[i]["use"]:
                        agents[i].sendCommand("use 0")
                        active[i]["use"] = 0
                    agents[i].sendCommand(f"{cmd} 1")
                    agents[i].sendCommand(f"{cmd} 0")
                elif cmd == "move" and val == 0.0:
                    # Stand still: only stop movement, don't cancel eating
                    agents[i].sendCommand("move 0")
                    active[i]["move"] = 0
                else:
                    # Continuous: send and track
                    agents[i].sendCommand(f"{cmd} {val}")
                    if cmd in active[i]:
                        active[i][cmd] = 1 if val != 0 else 0
            except Exception:
                pass

            brains[i].record_reward(reward)

        if not any_running:
            break

        # Agent think rate — sleep adjusted by game speed
        time.sleep(MS_PER_TICK / GAME_SPEED / 1000.0)

    # End of episode — compute stats, update brains, record
    ms_per_tick = MS_PER_TICK
    episode_stats = {}
    for i in range(NUM_AGENTS):
        total_reward = sum(brains[i]._ep_rewards)
        alive_sec = alive_ticks[i] * ms_per_tick / 1000.0
        total_sec = tick * ms_per_tick / 1000.0
        survival = alive_ticks[i] / max(tick, 1)

        episode_stats[AGENT_NAMES[i]] = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "episode": episode,
            "alive_ticks": alive_ticks[i],
            "total_ticks": tick,
            "alive_seconds": round(alive_sec, 1),
            "total_seconds": round(total_sec, 1),
            "survival": round(survival, 4),
            "died": dead[i],
            "food_eaten": food_eaten[i],
            "total_reward": round(total_reward, 1),
            "vocab_size": brains[i].next_block_id,
        }

        brains[i].update()
        brains[i].record_episode_stats(episode_stats[AGENT_NAMES[i]])

    return {
        "episode": episode,
        "ticks": tick,
        "agents": episode_stats,
    }


def main():
    args = parse_args()
    ports = [int(p) for p in args.ports.split(",")]

    ep_min = EPISODE_TIME_MS / 60000
    game_ms_per_tick = max(1, 50 // GAME_SPEED)
    agent_thinks_per_sec = 1000 // MS_PER_TICK
    input_desc = {
        1: "health + food + eating flag (3)",
        2: "health + food + eating + held item ID/count (5)",
        3: "health + food + eating + held + hotbar IDs/counts (23)",
        4: "vision + all above + movement flags (128)",
        5: "ALL 133 inputs",
    }
    action_desc = {
        1: "eat + still (2)",
        2: "eat + hotbar 1-3 + still (5)",
        3: "eat + hotbar 1-9 + still (11)",
        4: "eat + hotbar + walk + turn (15)",
        5: "eat + hotbar + all movement + attack + throw (21)",
        6: "ALL 23 actions",
    }
    print("=" * 60)
    print("  STAGE 1: Food in reach — learn to eat")
    print(f"  4 agents | 3-layer NN | world {GAME_SPEED}x speed | agent thinks every {MS_PER_TICK}ms")
    print(f"  Episode: {ep_min:.0f} min | game tick: {game_ms_per_tick}ms | agent: {agent_thinks_per_sec} decisions/sec")
    print(f"  Inputs  level {args.input_level}: {input_desc.get(args.input_level, '?')}")
    print(f"  Actions level {args.action_level}: {action_desc.get(args.action_level, '?')}")
    print(f"  Reward: +1 alive, {REWARD_DEATH:,.0f} death (LOCKED)")
    print("=" * 60)
    print()

    # Create brains — each agent gets its own
    # Input and action masks are separate — can be at different levels
    brains = []
    for i in range(NUM_AGENTS):
        name = AGENT_NAMES[i]
        brain = Brainstem(name=name, learning_rate=LEARNING_RATE,
                          input_level=args.input_level, action_level=args.action_level)

        # Auto-load from checkpoint if it exists
        load_dir = args.load_dir or args.load or SAVE_DIR
        agent_dir = os.path.join(load_dir, name)
        if os.path.exists(os.path.join(agent_dir, "weights.npz")):
            brain.load(agent_dir)
            brain.set_levels(args.input_level, args.action_level)
            print(f"  Loaded {name} (ep={brain.episodes_trained}, vocab={brain.next_block_id})")
        else:
            print(f"  {name} starting fresh")

        brains.append(brain)

    # Training loop
    all_stats = []
    for ep in range(1, args.episodes + 1):
        print(f"\n--- Episode {ep}/{args.episodes} ---")

        # Generate mission XML
        xml = mission_xml(time_limit_ms=EPISODE_TIME_MS,
                          ms_per_tick=max(1, MS_PER_TICK // GAME_SPEED))

        # Connect
        agents = connect_agents(ports, xml)
        if agents is None:
            print("Failed to connect. Make sure 4 Malmo clients are running.")
            print("Run: run\\launch_4_clients.bat")
            return

        # Setup: hard difficulty (starvation kills), fill hotbar with food
        time.sleep(0.5)
        try:
            agents[0].sendCommand("chat /difficulty hard")
        except Exception:
            pass
        time.sleep(0.2)

        # Fill all 9 hotbar slots with 10 of each food (limited supply)
        # Use agent 0 to issue all commands (it's the server host)
        # 1.11.2 uses minecraft: prefix for item IDs
        foods = [
            "minecraft:cooked_beef",
            "minecraft:cooked_chicken",
            "minecraft:cooked_fish",
            "minecraft:melon",
            "minecraft:carrot",
            "minecraft:baked_potato",
            "minecraft:apple",
            "minecraft:bread",
            "minecraft:cooked_porkchop",
        ]
        for name in AGENT_NAMES:
            try:
                agents[0].sendCommand(f"chat /clear {name}")
                time.sleep(0.1)
                for food in foods:
                    agents[0].sendCommand(f"chat /give {name} {food} 3")
                    time.sleep(0.05)
            except Exception:
                pass
        time.sleep(1.0)  # wait for all items to arrive
        # Run episode
        stats = run_episode(agents, brains, ep)
        all_stats.append(stats)

        # Print full episode report
        print(f"\n  Episode {ep} Results ({stats['ticks']:,} ticks)")
        print(f"  {'Agent':>5} | {'Survival':>8} | {'Alive':>10} | {'Died':>4} | {'Ate':>3} | {'Reward':>12} | {'Vocab':>5}")
        print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*10}-+-{'-'*4}-+-{'-'*3}-+-{'-'*12}-+-{'-'*5}")
        for name in AGENT_NAMES:
            s = stats['agents'][name]
            surv = s['survival'] * 100
            alive = f"{s['alive_seconds']:.0f}s/{s['total_seconds']:.0f}s"
            died = "DEAD" if s['died'] else "alive"
            print(f"  {name:>5} | {surv:>7.1f}% | {alive:>10} | {died:>4} | {s['food_eaten']:>3} | {s['total_reward']:>12,.0f} | {s['vocab_size']:>5}")

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
        first = all_stats[0]
        last = all_stats[-1]
        def avg_survival(s):
            return sum(s['agents'][n]['survival'] for n in AGENT_NAMES) / 4 * 100
        print(f"  Episodes: {len(all_stats)}")
        print(f"  First avg survival: {avg_survival(first):.1f}%")
        print(f"  Last  avg survival: {avg_survival(last):.1f}%")
        print(f"  Weights saved to: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
