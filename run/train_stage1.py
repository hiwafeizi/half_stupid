"""Stage 1 Training: Food in reach — learn to eat.

World: 16x16 grassy yard, daytime, fence walls, cake blocks nearby.
Agents: 4 (Adam, Eve, Cain, Abel), one life per episode.
Brain: reflex 529->32->23 (every tick) + context 160->64->23 (every 5 ticks).
Embedding: 2048 vocab x 4 dims, shared across all block/item/agent IDs.
Inputs: 118 IDs (100 vision + 9 hotbar + 1 held + 8 entities) -> 472 embed + 57 raw = 529.
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
from brain.brainstem import Brainstem, ACTIONS, ACTION_NAMES
import brain.brainstem as brainstem_module


# ═══════════════════════════════════════════════════════════════
#  CONFIG — change these freely
# ═══════════════════════════════════════════════════════════════
BASE_PORT = 10000
SAVE_DIR = "run/checkpoints/stage1"
EPISODE_TIME_MS = 3000000      # 50 minutes real time per episode
GAME_SPEED =8                # How fast the Minecraft WORLD runs vs real time.
                               # 1 = real time. 20 = world runs 20x faster.
                               # Hunger drains 20x faster, day passes 20x faster.
                               # Malmo MsPerTick = 50 / GAME_SPEED (lower = faster world).
                               # The agent's brain speed is NOT affected by this.

MS_PER_TICK = 100               # Agent think rate in GAME time (ms).
                               # 50 = 20 decisions per game second.
                               # At GAME_SPEED=20, real sleep = 50/20 = 2.5ms = 400 decisions/real sec.
                               # GAME_SPEED only affects real time, not game logic.
DEFAULT_EPISODES = 50          # default number of episodes
LEARNING_RATE = 0.03          # 0.1 for discovery, 0.03 for refinement, 0.01 for precision
GAMMA = 0.97                   # discount factor for REINFORCE returns
                               # 0.97 = punishes ~130 ticks (~6.5 sec) before death
                               # increase for later stages where long-term planning matters
MIN_EXPLORATION = 0.001       # minimum probability per enabled action
                               # prevents dead exploration but interrupts eating ~7.7% over 16 ticks
                               # lower = less interruption, higher = more exploration
CONTEXT_LR_SCALE = 1.5        # context layer learns at this fraction of LEARNING_RATE
                               # 0.5 = half speed, 0.2 = slow and steady

# ─── LOGGING FREQUENCY ───────────────────────────────────────
LIVE_JSON_EVERY = 100000          # overwrite live_<name>.json every N observations (0 = disabled)
PROBS_JSON_EVERY = 500         # append to probs_<name>.json every N observations (0 = disabled)

# ─── HOTBAR RANDOMIZATION ────────────────────────────────────
# Randomized each episode for generalization
HOTBAR_SLOTS_FILLED = (6, 9)   # min, max slots filled per episode (out of 9)
HOTBAR_UNEATABLE = (2, 4)      # min, max uneatable items mixed in
HOTBAR_POISON = (1, 3)         # min, max poison items mixed in (looks edible but hurts)
HOTBAR_ITEM_COUNT = (1, 5)     # min, max count per item

# ─── INPUT MASK LEVEL ─────────────────────────────────────────
# See brain/brainstem.py MASK_LEVELS for full details
# Level 1: health + food + eating flag (3 active)
# Level 2: + held item ID + held item count (8 active)
# Level 3: + 9 hotbar IDs + 9 hotbar counts (53 active)
# Level 4: + 100 vision (5x5x4) + 7 movement action flags (460 active)
# Level 5: + nearby entities (3 agents + 5 items) + x,y,z,yaw,pitch (529 active = all)
INPUT_MASK_LEVEL = 5

# ─── OUTPUT (ACTION) MASK LEVEL ──────────────────────────────
# See brain/brainstem.py ACTION_MASK_LEVELS for full details
# Level 1: eat + stand still (2 active)
# Level 2: + hotbar 1-3 (5 active)
# Level 3: + hotbar 4-9 (11 active)
# Level 4: + walk fwd/back + turn (15 active)
# Level 5: + strafe, jump, crouch, attack, throw (21 active)
# Level 6: + look up/down (23 active = all)
ACTION_MASK_LEVEL = 6

# ═══════════════════════════════════════════════════════════════
#  REWARD — DO NOT CHANGE. Only survival. See feedback_reward.md
# ═══════════════════════════════════════════════════════════════
REWARD_ALIVE = 1.0            # +1 per tick alive
REWARD_DEATH = -10000.0       # -10,000 on death


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

    # All agents must call startMission before Malmo begins any of them
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

    # Wait for all to start (with timeout so we don't hang forever)
    print("  Waiting for mission...", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < 120:
        states = [agents[i].getWorldState() for i in range(NUM_AGENTS)]
        if all(ws.has_mission_begun for ws in states):
            print(" Started!")
            return agents
        # Check for errors
        for i, ws in enumerate(states):
            if len(ws.errors) > 0:
                print(f"\n  ERROR on {AGENT_NAMES[i]}: {ws.errors[0].text}")
                return None
        time.sleep(0.3)
        print(".", end="", flush=True)

    # Timeout — show which agents didn't start
    for i in range(NUM_AGENTS):
        ws = agents[i].getWorldState()
        if not ws.has_mission_begun:
            print(f"\n  {AGENT_NAMES[i]} never started (port {ports[i]})")
    print("\n  TIMEOUT. Restart Malmo clients and retry.")
    return None


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
    tick = 0                      # actual observations processed (across all agents)
    loop_count = 0                 # raw loop iterations (for internal use)
    alive_ticks = [0] * NUM_AGENTS

    # Clear live logs for this episode
    os.makedirs(SAVE_DIR, exist_ok=True)
    for name in AGENT_NAMES:
        p = os.path.join(SAVE_DIR, f"live_{name}.json")
        if os.path.exists(p):
            os.remove(p)
    dead = [False] * NUM_AGENTS
    food_eaten = [0] * NUM_AGENTS
    prev_food = [20.0] * NUM_AGENTS
    obs_count = [0] * NUM_AGENTS   # per-agent observation count
    prob_logs = {name: [] for name in AGENT_NAMES}
    # Track active continuous commands per agent
    active = [{"use": 0, "move": 0, "strafe": 0, "turn": 0,
               "pitch": 0, "jump": 0, "crouch": 0, "attack": 0}
              for _ in range(NUM_AGENTS)]

    from threading import Thread

    def _agent_tick(i, obs, tick_num):
        """One agent's full tick: encode, forward, choose action. Runs in thread."""
        health = obs.get("Life", 0)
        food_level = obs.get("Food", 0)

        if health <= 0:
            dead[i] = True
            brains[i].record_reward(REWARD_DEATH)
            print(f"    {AGENT_NAMES[i]} DIED at tick {tick_num} (alive {alive_ticks[i]} ticks)")
            return None

        alive_ticks[i] += 1

        if food_level > prev_food[i]:
            food_eaten[i] += 1
        prev_food[i] = food_level

        brains[i].record_reward(REWARD_ALIVE)

        grid = obs.get("view5x5", [])
        action_idx = brains[i].choose_action(obs, grid, active[i])
        return action_idx

    while True:
        loop_count += 1
        any_running = False

        # Step 1: Poll all agents for observations (sequential — Malmo requires this)
        observations = [None] * NUM_AGENTS
        for i in range(NUM_AGENTS):
            if dead[i]:
                continue
            ws = agents[i].getWorldState()
            if not ws.is_mission_running:
                continue
            any_running = True
            if ws.number_of_observations_since_last_state > 0:
                observations[i] = json.loads(ws.observations[-1].text)

        if not any_running:
            break

        # Count actual observations
        got_obs = any(o is not None for o in observations)
        if got_obs:
            tick += 1
        for i in range(NUM_AGENTS):
            if observations[i] is not None:
                obs_count[i] += 1

        # Step 2: Run all agent brains in parallel threads
        results = [None] * NUM_AGENTS
        threads = []
        for i in range(NUM_AGENTS):
            if observations[i] is None or dead[i]:
                continue
            t = Thread(target=lambda idx=i: results.__setitem__(idx, _agent_tick(idx, observations[idx], tick)))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        # Step 3: Apply hunger effect (sequential — Malmo commands)
        if tick % HUNGER_DRAIN_EVERY == 0:
            for i in range(NUM_AGENTS):
                if dead[i] or observations[i] is None:
                    continue
                try:
                    agents[i].sendCommand(f"chat /effect {AGENT_NAMES[i]} 17 5 127 true")
                except Exception:
                    pass

        # Step 4: Send actions + write debug JSON (sequential)
        for i in range(NUM_AGENTS):
            action_idx = results[i]
            if action_idx is None:
                continue

            # Live JSON
            if LIVE_JSON_EVERY and obs_count[i] % LIVE_JSON_EVERY == 0:
                brain = brains[i]
                ids = brain._last_ids
                rf = brain._last_raw
                pr = brain._last_probs.tolist()
                x465 = getattr(brain, '_last_x', None)
                # Only the values the AI actually receives (non-zero after masking)
                INPUT_LABELS = {}
                for j in range(110):
                    for d in range(4):
                        INPUT_LABELS[j * 4 + d] = f"embed_{j}_d{d}"
                RAW_LABELS = [
                    "health", "food", "x", "y", "z", "yaw", "pitch",
                    "s1_ct", "s2_ct", "s3_ct", "s4_ct", "s5_ct",
                    "s6_ct", "s7_ct", "s8_ct", "s9_ct",
                    "held_ct", "eating", "moving", "strafing", "turning",
                    "pitching", "jumping", "crouching", "attacking",
                    "a1_here", "a1_rx", "a1_rz", "a1_hp",
                    "a2_here", "a2_rx", "a2_rz", "a2_hp",
                    "a3_here", "a3_rx", "a3_rz", "a3_hp",
                    "i1_here", "i1_rx", "i1_rz", "i1_qty",
                    "i2_here", "i2_rx", "i2_rz", "i2_qty",
                    "i3_here", "i3_rx", "i3_rz", "i3_qty",
                    "i4_here", "i4_rx", "i4_rz", "i4_qty",
                    "i5_here", "i5_rx", "i5_rz", "i5_qty",
                ]
                for j in range(len(RAW_LABELS)):
                    INPUT_LABELS[472 + j] = RAW_LABELS[j]

                ai_input = {}
                if x465 is not None:
                    for j, v in enumerate(x465):
                        if abs(v) > 1e-6:
                            ai_input[INPUT_LABELS.get(j, str(j))] = round(float(v), 4)

                reverse_vocab = {v: k for k, v in brain.block_vocab.items()}
                held_id = int(ids[109])
                snapshot = {
                    "tick": tick, "obs": obs_count[i],
                    "alive_seconds": round(alive_ticks[i] * MS_PER_TICK / 1000.0, 1),
                    "input_level": brain.input_level,
                    "action_level": brain.action_level,
                    "enabled_actions": int(brain.action_mask.sum()),
                    "held": reverse_vocab.get(held_id, "EMPTY"),
                    "eating": int(rf[17]),
                    "ai_input": ai_input,
                    "output": {ACTION_NAMES[j]: round(p, 6) for j, p in enumerate(pr) if brain.action_mask[j] > 0},
                    "chosen": ACTION_NAMES[action_idx],
                }
                debug_path = os.path.join(SAVE_DIR, f"live_{AGENT_NAMES[i]}.json")
                with open(debug_path, "a") as f:
                    f.write(json.dumps(snapshot) + "\n")

            # Probs log
            if PROBS_JSON_EVERY and obs_count[i] % PROBS_JSON_EVERY == 0:
                brain = brains[i]
                pr = brain._last_probs.tolist()
                rf = brain._last_raw
                ids = brain._last_ids
                prob_entry = {
                    "tick": tick, "obs": obs_count[i],
                    "input_level": brain.input_level,
                    "action_level": brain.action_level,
                    "ids": [int(v) for v in ids],
                    "raw": [round(float(v), 2) for v in rf],
                    "probs": {ACTION_NAMES[j]: round(p, 6) for j, p in enumerate(pr) if brain.action_mask[j] > 0},
                    "chosen": ACTION_NAMES[action_idx],
                }
                prob_logs[AGENT_NAMES[i]].append(prob_entry)
                prob_path = os.path.join(SAVE_DIR, f"probs_{AGENT_NAMES[i]}.json")
                with open(prob_path, "w") as f:
                    json.dump(prob_logs[AGENT_NAMES[i]], f, indent=2)

            # Execute action
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
                    # Stand still: stop movement, don't cancel eating
                    agents[i].sendCommand("move 0")
                    active[i]["move"] = 0
                elif cmd in ("use", "crouch"):
                    # Continuous: eat, crouch, turn stay held until changed
                    agents[i].sendCommand(f"{cmd} {val}")
                    if cmd in active[i]:
                        active[i][cmd] = 1 if val != 0 else 0
                else:
                    # Pulse: everything else fires once then resets
                    # move, strafe, turn, pitch, jump, attack
                    agents[i].sendCommand(f"{cmd} {val}")
                    agents[i].sendCommand(f"{cmd} 0")
                    if cmd in active[i]:
                        active[i][cmd] = 0
            except Exception:
                pass

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
POISON_ITEMS = [
    "minecraft:spider_eye",         # gives poison effect
    "minecraft:rotten_flesh",       # gives hunger effect (food drains faster)
    "minecraft:poisonous_potato",   # 60% chance of poison
    "minecraft:raw_chicken",        # 30% chance of hunger effect
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
        2: "health + food + eating + held item ID/count (8)",
        3: "health + food + eating + held + hotbar IDs/counts (53)",
        4: "vision(100) + all above + movement flags (460)",
        5: "ALL 529: + entities(3 agents + 5 items) + position",
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
    # Apply context layer learning rate scale before creating brains
    brainstem_module.CONTEXT_LR_SCALE = CONTEXT_LR_SCALE

    brains = []
    for i in range(NUM_AGENTS):
        name = AGENT_NAMES[i]
        brain = Brainstem(name=name, learning_rate=LEARNING_RATE,
                          input_level=args.input_level, action_level=args.action_level,
                          min_exploration=MIN_EXPLORATION)

        # Auto-load from checkpoint if it exists
        load_dir = args.load_dir or args.load or SAVE_DIR
        agent_dir = os.path.join(load_dir, name)
        if os.path.exists(os.path.join(agent_dir, "weights.pt")):
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

        # Connect (retry on timeout — Malmo sometimes needs a kick)
        agents = None
        for retry in range(3):
            agents = connect_agents(ports, xml)
            if agents is not None:
                break
            print(f"  Retry {retry + 1}/3 — regenerating mission...")
            time.sleep(5)
            xml = mission_xml(time_limit_ms=EPISODE_TIME_MS,
                              ms_per_tick=max(1, MS_PER_TICK // GAME_SPEED))
        if agents is None:
            print("  Failed 3 times. Skipping episode.")
            continue

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
                num_poison = min(random.randint(*HOTBAR_POISON), num_slots - num_uneatable)
                num_food = num_slots - num_uneatable - num_poison

                food_picks = random.sample(FOOD_ITEMS, min(num_food, len(FOOD_ITEMS)))
                junk_picks = random.sample(UNEATABLE_ITEMS, min(num_uneatable, len(UNEATABLE_ITEMS)))
                poison_picks = random.sample(POISON_ITEMS, min(num_poison, len(POISON_ITEMS)))
                items = food_picks + junk_picks + poison_picks
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
