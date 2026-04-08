"""Stage 1 Training: Food in reach — learn to eat.

World: 16x16 grassy yard, daytime, fence walls, cake blocks nearby.
Agents: 4 (Adam, Eve, Cain, Abel), one life per episode.
Brain: 465 -> 32 -> 23 (embedding + 1 hidden layer, REINFORCE).
Embedding: 2048 vocab x 4 dims, shared across all block/item IDs.
Inputs: 110 IDs (100 vision + 9 hotbar + 1 held) -> 440 embed + 25 raw = 465.
Reward: ONLY +1 alive, death penalty. Nothing else. Ever.
Hunger: /effect hunger drains food bar, starvation kills on hard difficulty.
Hotbar: randomized each episode (random foods, uneatable items, random slots).
Masking: input (1-5) and action (1-6) levels configurable.
Saves: per-agent weights + embeddings, vocab, full episode history. Auto-loads.
Debug: live_<name>.json every ~1 sec, probs_<name>.json every 20 ticks.

Usage:
    python run/start.py --skip-launch --episodes 10
    python run/start.py --skip-launch --input-level 3 --action-level 3
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
from brain.brainstem import Brainstem, ACTIONS, NUM_ACTIONS, ACTION_NAMES, NUM_ID_INPUTS
import numpy as np


# ═══════════════════════════════════════════════════════════════
#  CONFIG — change these freely
# ═══════════════════════════════════════════════════════════════
BASE_PORT = 10000
SAVE_DIR = "run/checkpoints/stage1"
EPISODE_TIME_MS = 3000000      # 50 minutes real time per episode
GAME_SPEED =5                # How fast the Minecraft WORLD runs vs real time.
                               # 1 = real time. 20 = world runs 20x faster.
                               # Hunger drains 20x faster, day passes 20x faster.
                               # Malmo MsPerTick = 50 / GAME_SPEED (lower = faster world).
                               # The agent's brain speed is NOT affected by this.

MS_PER_TICK = 50               # Agent think rate in GAME time (ms).
                               # 50 = 20 decisions per game second.
                               # At GAME_SPEED=20, real sleep = 50/20 = 2.5ms = 400 decisions/real sec.
                               # GAME_SPEED only affects real time, not game logic.
DEFAULT_EPISODES = 50          # default number of episodes
LEARNING_RATE = 0.03          # 0.1 for discovery, 0.03 for refinement, 0.01 for precision
GAMMA = 0.97                   # discount factor for REINFORCE returns
                               # 0.97 = punishes ~130 ticks (~6.5 sec) before death
                               # increase for later stages where long-term planning matters
MIN_EXPLORATION = 0.0002       # minimum probability per enabled action
                               # prevents dead exploration but interrupts eating ~7.7% over 16 ticks
                               # lower = less interruption, higher = more exploration

# ─── HOTBAR RANDOMIZATION ────────────────────────────────────
# Randomized each episode for generalization
HOTBAR_SLOTS_FILLED = (9, 9)   # min, max slots filled per episode (out of 9)
HOTBAR_UNEATABLE = (0, 0)      # min, max uneatable items mixed in
HOTBAR_ITEM_COUNT = (1, 5)     # min, max count per item

# ─── INPUT MASK LEVEL ─────────────────────────────────────────
# See brain/brainstem.py MASK_LEVELS for full details
# Level 1: health + food + eating flag (3 active)
# Level 2: + held item ID + held item count (5 active)
# Level 3: + 9 hotbar IDs + 9 hotbar counts (23 active)
# Level 4: + 100 vision (5x5x4) + 7 movement action flags (130 active)
# Level 5: + x,y,z + yaw + pitch (135 active = all)
INPUT_MASK_LEVEL = 2

# ─── OUTPUT (ACTION) MASK LEVEL ──────────────────────────────
# See brain/brainstem.py ACTION_MASK_LEVELS for full details
# Level 1: eat + stand still (2 active)
# Level 2: + hotbar 1-3 (5 active)
# Level 3: + hotbar 4-9 (11 active)
# Level 4: + walk fwd/back + turn (15 active)
# Level 5: + strafe, jump, crouch, attack, throw (21 active)
# Level 6: + look up/down (23 active = all)
ACTION_MASK_LEVEL = 1

# ═══════════════════════════════════════════════════════════════
#  REWARD — DO NOT CHANGE. Only survival. See feedback_reward.md
# ═══════════════════════════════════════════════════════════════
REWARD_ALIVE = 1.0            # +1 per tick alive
REWARD_DEATH = -2000.0       # -4,000 on death


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
    prob_logs = {name: [] for name in AGENT_NAMES}  # probs only, every 20 ticks
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
            grid = obs.get("view5x5", [])
            action_idx = brains[i].choose_action(obs, grid, active[i])

            # Save live debug snapshot every ~1 second
            # Shows EXACTLY what the AI sees — raw IDs, raw floats, embeddings, output probs
            if tick % 500 == i:
                brain = brains[i]
                ids = brain._ep_id_indices[-1] if brain._ep_id_indices else np.zeros(NUM_ID_INPUTS, dtype=np.int32)
                net_input = brain._ep_inputs[-1].tolist() if brain._ep_inputs else []
                probs = brain._ep_probs[-1].tolist() if brain._ep_probs else []

                # Get raw floats from _encode (re-extract from obs)
                _, raw_floats = brain._encode(obs, grid, active[i])

                ms_tick = MS_PER_TICK
                snapshot = {
                    "agent": AGENT_NAMES[i],
                    "episode": episode,
                    "tick": tick,
                    "alive_ticks": alive_ticks[i],
                    "alive_seconds": round(alive_ticks[i] * ms_tick / 1000.0, 1),
                    "episode_seconds": round(tick * ms_tick / 1000.0, 1),

                    # === WHAT AI SEES: raw IDs (before embedding) ===
                    "ids": {
                        "vision_below": [int(ids[j]) for j in range(0, 25)],
                        "vision_floor": [int(ids[j]) for j in range(25, 50)],
                        "vision_eye": [int(ids[j]) for j in range(50, 75)],
                        "vision_above": [int(ids[j]) for j in range(75, 100)],
                        "hotbar": [int(ids[100+j]) for j in range(9)],
                        "held": int(ids[109]),
                    },

                    # === WHAT AI SEES: raw floats (no embedding) ===
                    "raw": {
                        "health": float(raw_floats[0]),
                        "food": float(raw_floats[1]),
                        "x": round(float(raw_floats[2]), 1),
                        "y": round(float(raw_floats[3]), 1),
                        "z": round(float(raw_floats[4]), 1),
                        "yaw": round(float(raw_floats[5]), 1),
                        "pitch": round(float(raw_floats[6]), 1),
                        "hotbar_counts": [int(raw_floats[7+j]) for j in range(9)],
                        "held_count": int(raw_floats[16]),
                        "eating": int(raw_floats[17]),
                        "moving": int(raw_floats[18]),
                        "strafing": int(raw_floats[19]),
                        "turning": int(raw_floats[20]),
                        "pitching": int(raw_floats[21]),
                        "jumping": int(raw_floats[22]),
                        "crouching": int(raw_floats[23]),
                        "attacking": int(raw_floats[24]),
                    },

                    # === WHAT AI SEES: 465-float network input (embedded + raw) ===
                    "network_input_465": [round(v, 4) for v in net_input] if net_input else [],

                    # === AI OUTPUT: action probabilities ===
                    "output": {
                        "probs": {ACTION_NAMES[j]: round(p, 8)
                                  for j, p in enumerate(probs) if p > 0.000001},
                        "chosen": {
                            "index": action_idx,
                            "name": ACTION_NAMES[action_idx],
                            "command": f"{ACTIONS[action_idx][0]} {ACTIONS[action_idx][1]}",
                        },
                    },

                    # === MASKS ===
                    "levels": {
                        "input": brain.input_level,
                        "action": brain.action_level,
                        "active_id_inputs": int(brain.id_mask.sum()),
                        "active_raw_inputs": int(brain.raw_mask.sum()),
                        "active_actions": int(brain.action_mask.sum()),
                    },

                    # === DICTIONARIES (human-readable, separate from AI data) ===
                    "dict_blocks": {str(v): k for k, v in brain.block_vocab.items()},
                    "dict_actions": {str(j): ACTION_NAMES[j] for j in range(NUM_ACTIONS)},
                }
                live_logs[AGENT_NAMES[i]].append(snapshot)
                debug_path = os.path.join(SAVE_DIR, f"live_{AGENT_NAMES[i]}.json")
                os.makedirs(SAVE_DIR, exist_ok=True)
                with open(debug_path, "w") as f:
                    json.dump(live_logs[AGENT_NAMES[i]], f, indent=2)

            # Probs-only log every 1000 ticks (independent of live snapshot)
            if tick % 1000 == 0:
                brain = brains[i]
                p_raw = brain._ep_probs[-1].tolist() if brain._ep_probs else []

                if p_raw:
                    # Show raw floats (unmasked) — what the AI actually has
                    AN = ACTION_NAMES
                    p_ids = brain._ep_id_indices[-1] if brain._ep_id_indices else []
                    _, p_raw_floats = brain._encode(obs, grid, active[i])

                    # Build readable active inputs
                    RAW_NAMES = [
                        "health", "food", "x", "y", "z", "yaw", "pitch",
                        "slot1_ct", "slot2_ct", "slot3_ct", "slot4_ct", "slot5_ct",
                        "slot6_ct", "slot7_ct", "slot8_ct", "slot9_ct",
                        "held_ct", "eating", "moving", "strafing", "turning",
                        "pitching", "jumping", "crouching", "attacking",
                    ]
                    active_raw = {RAW_NAMES[j]: round(float(p_raw_floats[j]), 1)
                                  for j in range(len(RAW_NAMES))
                                  if brain.raw_mask[j] > 0}
                    active_ids = {}
                    if brain.id_mask[109] > 0 and len(p_ids) > 109:
                        active_ids["held_id"] = int(p_ids[109])
                    for s in range(9):
                        if brain.id_mask[100 + s] > 0 and len(p_ids) > 100 + s:
                            active_ids[f"slot{s+1}_id"] = int(p_ids[100 + s])

                    prob_entry = {
                        "tick": tick,
                        "alive_seconds": round(alive_ticks[i] * MS_PER_TICK / 1000.0, 1),
                        "inputs": {**active_raw, **active_ids},
                        "probs": {AN[j]: round(p, 8)
                                  for j, p in enumerate(p_raw) if p > 0.000001},
                        "chosen": AN[action_idx],
                    }
                    prob_logs[AGENT_NAMES[i]].append(prob_entry)
                    prob_path = os.path.join(SAVE_DIR, f"probs_{AGENT_NAMES[i]}.json")
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    with open(prob_path, "w") as f:
                        json.dump(prob_logs[AGENT_NAMES[i]], f, indent=2)

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

        brains[i].update(gamma=GAMMA)
        brains[i].record_episode_stats(episode_stats[AGENT_NAMES[i]])

    return {
        "episode": episode,
        "ticks": tick,
        "agents": episode_stats,
    }


# ─── ITEM POOLS (names only, settings above) ────────────────
FOOD_ITEMS = [
    "minecraft:cooked_beef", "minecraft:cooked_chicken", "minecraft:cooked_fish",
    "minecraft:cooked_porkchop", "minecraft:cooked_mutton", "minecraft:cooked_rabbit",
    "minecraft:melon", "minecraft:carrot", "minecraft:baked_potato",
    "minecraft:apple", "minecraft:bread", "minecraft:cookie",
    "minecraft:pumpkin_pie", "minecraft:golden_apple",
    "minecraft:mushroom_stew", "minecraft:beetroot_soup",
]
UNEATABLE_ITEMS = [
    "minecraft:stone", "minecraft:stick", "minecraft:cobblestone", "minecraft:dirt",
    "minecraft:iron_ingot", "minecraft:gold_ingot", "minecraft:diamond", "minecraft:coal",
    "minecraft:bone", "minecraft:feather", "minecraft:string", "minecraft:arrow",
    "minecraft:wooden_sword", "minecraft:wooden_pickaxe", "minecraft:leather", "minecraft:paper",
]


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
        4: "vision(100) + all above + movement flags (130)",
        5: "ALL 135 inputs",
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
                          input_level=args.input_level, action_level=args.action_level,
                          min_exploration=MIN_EXPLORATION)

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
        # Malmo MsPerTick: 50 = normal speed. Lower = faster.
        # GAME_SPEED=20 → MsPerTick=2 (was working before)
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
            agents[0].sendCommand("chat /gamerule sendCommandFeedback false")
            time.sleep(0.1)
            agents[0].sendCommand("chat /difficulty hard")
        except Exception:
            pass
        time.sleep(0.2)

        # Randomized hotbar setup each episode for generalization:
        # - Random food order and positions
        # - Random counts (1-5 per item)
        # - Some slots empty
        # - Some slots have uneatable items (stone, stick)
        import random

        for name in AGENT_NAMES:
            try:
                agents[0].sendCommand(f"chat /clear {name}")
                time.sleep(0.1)

                num_slots = random.randint(*HOTBAR_SLOTS_FILLED)
                num_uneatable = min(random.randint(*HOTBAR_UNEATABLE), num_slots)
                num_food = num_slots - num_uneatable

                food_picks = random.sample(FOOD_ITEMS, min(num_food, len(FOOD_ITEMS)))
                junk_picks = random.sample(UNEATABLE_ITEMS, min(num_uneatable, len(UNEATABLE_ITEMS)))
                items = food_picks + junk_picks
                random.shuffle(items)

                # Place items in random slot positions using /replaceitem
                all_slots = list(range(9))
                random.shuffle(all_slots)
                for idx, item in enumerate(items):
                    slot = all_slots[idx]
                    count = random.randint(*HOTBAR_ITEM_COUNT)
                    agents[0].sendCommand(
                        f"chat /replaceitem entity {name} slot.hotbar.{slot} {item} {count}")
                    time.sleep(0.03)
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

        # Save best model per agent (highest alive_seconds)
        for i, name in enumerate(AGENT_NAMES):
            alive_sec = stats['agents'][name]['alive_seconds']
            best_file = os.path.join(SAVE_DIR, name, "best_seconds.txt")

            # Read current best
            current_best = 0.0
            if os.path.exists(best_file):
                with open(best_file) as f:
                    try:
                        current_best = float(f.read().strip())
                    except ValueError:
                        current_best = 0.0

            if alive_sec > current_best:
                # New best — save weights separately
                best_dir = os.path.join(SAVE_DIR, name, "best")
                brains[i].save(best_dir)
                with open(best_file, "w") as f:
                    f.write(str(alive_sec))
                print(f"  ** {name} NEW BEST: {alive_sec:.0f}s (was {current_best:.0f}s) **")

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
