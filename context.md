1. High level overview
2. World and agent assumptions
3. Detailed module design
4. Data formats
5. Brain loop and compute schedule
6. Training strategy

---

## 1. High level overview

You are building an embodied agent with:

* Multimodal perception (vision + sound + internal state)
* Three prefrontal subsystems
  * Reflex module (tiny neural, minimal input, highest priority)
  * Fast PFC (cheap, heuristic like decisions)
  * Reflective PFC (RL policy, uses memory and rewards)
  * Planning PFC (slower, using world model and spatial memory)
* Memory system
  * Short term memory (STM)
  * Long term memory (LTM, value tagged)
  * Spatial memory (map with reward info)
* Communication
  * Agent can emit 5 vocal symbols (A, B, C, D, E)
  * Agents can hear both Minecraft sounds and these vocal symbols
* Reward system focused on survival
  * Strong reward for staying alive
  * Smaller rewards or penalties for food, pain, social behavior, novelty

Goal: agents should learn to survive, navigate, and exhibit emergent social and communication behavior.

---

## 2. World and agent assumptions

You can tweak this later, but the design assumes:

* Discrete time steps (ticks), based on Minecraft or a grid world
* Agent has a position and direction
* Environment has:
  * Safe tiles
  * Dangerous tiles (lava, cliffs, hostile mobs etc.)
  * Food or resource tiles
  * Other agents
  * Minecraft sound events available from the API

Action space example:

* Move: up, down, left, right
* Look: rotate left, right
* Interact: grab, use, eat
* Vocalize: say A, B, C, D, or E
* Wait or rest

You can extend this but try to keep it limited at first.

---

## 3. Detailed module design

### 3.1 Sensory layer

**Inputs:**

* Vision
  * Encoded representation of nearby tiles (for example a 7x7 grid around the agent)
  * Types: empty, wall, lava, cliff, food, agent, etc.
* Sound
  * Last N audio events within radius R
  * Each event: type (Minecraft sound id), distance, direction, source (agent or environment), and optional letter if it is a vocal symbol
* Internal state
  * Health
  * Energy / hunger
  * Fear level
  * Pain level
  * Curiosity level

**Output:** a state vector that is fed into different brain regions, with subsets of it going to specific modules.

You can implement this as a function:

```python
def sense_environment(env, agent_id) -> Dict:
    return {
        "vision_patch": ...,
        "sounds": [...],
        "internal": {...},
        "timestamp": t
    }
```

This is design logic, not a scientific claim, so you should fact check only if you want theoretical background.

---

### 3.2 Memory system

You have three memory stores: STM, LTM, Spatial.

#### 3.2.1 Short term memory (STM)

Lifetime: a few seconds of simulated time.

Stores:

* Recent events (with immediate rewards)
* Short state trace
* Last actions

Example structure:

```python
STM = {
    "recent_events": [
        {
            "event_type": "see_lava",
            "dist": 1,
            "action": "jump_back",
            "reward": -0.2,
            "time": 2931
        },
        {
            "event_type": "heard_vocal",
            "letter": "A",
            "from_agent": 12,
            "reward": 0.01,
            "time": 2932
        }
    ],
    "state_trace": [
        {"state_vec": [...], "reward": 0.0, "time": 2929},
        {"state_vec": [...], "reward": 0.02, "time": 2930}
    ],
    "recent_actions": ["walk_east", "walk_east", "look_north"]
}
```

STM is updated every tick and truncated to a max length.

#### 3.2.2 Long term memory (LTM)

Lifetime: long. Can be saved to disk.

Key components:

* Action statistics
* State action value approximations
* Learned vocal meanings
* Danger and safety statistics

Example:

```python
LTM = {
    "action_stats": {
        "jump_back": {
            "avg_reward": 0.42,
            "times_used": 52,
            "success_rate": 0.78
        },
        "say_A": {
            "avg_reward": -0.05,
            "times_used": 40,
            "success_rate": 0.20
        }
    },
    "state_action_pairs": {
        "hash_abc123": {
            "action": "run",
            "total_reward": 2.3,
            "n": 14,
            "utility": 0.164
        }
    },
    "vocal_meanings": {
        "A": {"avg_reward_context": -0.3, "confidence": 0.7},
        "B": {"avg_reward_context": 0.15, "confidence": 0.4}
    },
    "danger_tags": {
        "lava": {"avg_reward": -0.6, "visits": 30},
        "cliff": {"avg_reward": -0.5, "visits": 20}
    }
}
```

Average reward can be updated incrementally:

```python
def update_running_average(old_avg, reward, n):
    return old_avg + (reward - old_avg) / (n + 1)
```

This formula is standard in online averaging, but you should still fact check if you want an external reference.

LTM is periodically saved to disk as JSON or in a small database.

#### 3.2.3 Spatial memory

Stores map-like info with value tags.

```python
SPATIAL = {
    "(12,5)": {
        "tile": "grass",
        "reward_sum": 0.1,
        "visits": 30,
        "avg_reward": 0.003
    },
    "(13,5)": {
        "tile": "lava",
        "reward_sum": -15.0,
        "visits": 30,
        "avg_reward": -0.5
    }
}
```

When the agent leaves or visits a tile, it updates the reward statistics.

---

### 3.3 Reflex module (tiny neural)

Goal: fast reaction with minimal information.

**Inputs:** very small subset

* Local 3x3 tile types around the agent
* Closest danger distance (lava, cliff, hostile mob)
* Immediate sound types near by (like creeper hiss, explosion, your vocal A etc.)
* Fall velocity
* Current health

You should compress these into a very small vector, like 8 to 16 floats/bools.

**Network:**

* Input: 8 to 16 units
* Hidden: 2 to 4 ReLU units
* Output: 3 to 5 reflex actions
  * do nothing
  * jump back
  * stop movement
  * dodge sideways
  * crouch

Use softmax over outputs and either pick argmax or sample.

**Priority:** highest. If the highest probability reflex is above threshold, it overrides everything else for that tick.

**Learning:** optional and minimal. For example:

* On death, slightly reinforce reflexes that would have avoided that state, or punish reflexes that led into danger.
* At first you can hardcode weights or train offline in simple scenarios.

This is design logic, so you should fact check only if you want neuroscience or RL backing.

---

### 3.4 Fast PFC

Goal: slightly smarter, still cheap decision making.

**Inputs:**

* Compressed vision representation
* A bit of STM (last actions, recent danger direction)
* Internal state (hunger, fear, pain)

Network:

* Input: maybe 32 to 64 units
* Hidden: 32 units
* Output: full action space (move, interact, speak, wait)

Behavior:

* Runs every tick or every second tick
* Used when reflex does not fire
* Can be partially rule based at first (if hungry and food is near then go to food etc.)

This acts like a heuristic controller, giving the agent basic functionality before RL training converges.

---

### 3.5 Reflective PFC (RL policy network)

Goal: real learning and adaptation.

**Inputs:**

* Full state vector
  * processed vision
  * processed sound
  * internal state
  * summary of STM
  * summaries from LTM (for example risk estimate for current tile, estimated quality of candidate actions)

**Network:**

* Input: maybe 64 to 256 units (depending on compression)
* 1 or 2 hidden layers, 64 to 128 units each
* Outputs:
  * Policy: probability distribution over all actions
  * Optional value head: estimated value of current state

You can use standard RL algorithms like PPO or A2C here. These are widely used in practice, but for any specific algorithmic guarantee you should fact check with RL textbooks or papers.

Reflective PFC runs less frequently than fast PFC, for example every 5 ticks.

Its output is blended with fast PFC, or used to override it when confidence is high.

---

### 3.6 Planning PFC (world model based)

Goal: look ahead and choose actions using simulated outcomes.

Components:

1. World model network
   * Input: current state vector and candidate action
   * Output: predicted next state features and predicted reward (or change in reward)
2. Planner logic
   * For each candidate action (or a sample of them)
     * Use world model to roll a few steps into the future
     * Sum predicted rewards and use LTM and spatial memory to adjust risk estimates
   * Pick the action with the best predicted outcome

You can run this:

* Rarely (for example every 20 ticks)
* Or only when:
  * No immediate danger
  * The agent is at a decision point (fork in a path, new area etc.)

Planning PFC does not need to be large. The world model can be a small MLP. This is design based on common model based RL ideas, which you should fact check if you want strong theoretical grounding.

---

### 3.7 Emotion and reward shaping

Internal emotional variables:

* fear
* hunger
* curiosity
* pain

Update them each tick based on environment and internal state. Then use them in two places:

1. Reward
   * survival: +1 per step alive
   * food found: +0.2
   * damage taken: negative reward relative to pain
   * social vocalization near another agent: +0.01
   * novelty (first time in a tile or unexpected event): small positive reward
2. Decision weighting
   * Fear increases weighting of danger avoidance in action selection
   * Hunger pushes agent toward food seeking
   * Curiosity increases chance of exploring unknown tiles

All of this is design and you should fact check only if you want to see how it compares to standard reward shaping strategies.

---

### 3.8 Communication module (5 letters)

Agent can emit one of 5 symbols: A, B, C, D, E. Under the hood these map to specific Minecraft sounds or note block pitches.

**Production:**

* Treated as actions from the PFC networks
* Stored in STM and LTM as events with rewards

**Perception:**

* Transform raw Minecraft sound events into symbolic representation when they correspond to your 5 letters
* For example, map specific note block pitch to letter A

**Learning meaning:**

* In LTM, for each letter track: average reward context and success rate
* Example:
  * If hearing A is often followed by negative reward, A will be interpreted as danger
  * If hearing B before food often leads to positive reward, B becomes a food signal

No explicit semantics are hardcoded. The mapping emerges through reward association.

Again this is a design choice inspired by emergent communication work in AI, which you should fact check if you want research references.

---

## 4. Data formats summary

You can keep a simple `AgentBrainState` structure in code:

```python
class AgentBrainState:
    def __init__(self):
        self.stm = {...}
        self.ltm = {...}
        self.spatial = {...}
        self.internal_state = {...}
        self.last_action = None
        self.last_reward = 0.0
        self.time = 0
```

And a `Perception` structure:

```python
class Perception:
    def __init__(self, vision_patch, sounds, internal, time):
        self.vision_patch = vision_patch
        self.sounds = sounds
        self.internal = internal
        self.time = time
```

Where sounds and vision have concrete formats you decide.

---

## 5. Brain loop and compute schedule

Assume 20 ticks per second.

**Every tick:**

1. Sense environment
2. Update STM with new perception and last reward
3. Update internal emotional state
4. Run reflex network
   * If reflex confidence above threshold: output reflex action and skip lower modules
5. Run fast PFC
   * Propose action

**Every 5 ticks:**

6. Run reflective PFC
   * Compute policy and value
   * Optionally adjust or replace fast PFC decision

**Every 20 ticks:**

7. Run planning PFC
   * Evaluate candidate actions with world model and spatial memory
   * If planning confidence high: override previous action

**After action is decided:**

8. Execute action in environment
9. Receive reward
10. Update STM, LTM, SPATIAL with reward information
11. If tick count hits save interval: write LTM and SPATIAL to disk

This schedule is design based on balancing compute and responsiveness. You should fact check if you want to know typical tick rates for similar systems in research or games.

---

## 6. Training strategy

Recommended phases:

1. Phase 1: Reflex only
   * Implement reflex with hand tuned weights or simple rules
   * Ensure the agent does not constantly die immediately
2. Phase 2: Fast PFC with heuristics
   * Add rule based behavior for food seeking, basic movement
   * Confirm agent can explore and eat
3. Phase 3: RL for reflective PFC
   * Freeze reflex and heuristics
   * Train policy network to improve over heuristics using PPO or similar
   * Make sure rewards are not too sparse
4. Phase 4: Memory value tagging
   * Start logging STM → LTM and SPATIAL with reward statistics
   * Use statistics to bias policy and planning
5. Phase 5: Planning PFC
   * Train a world model on recorded transitions
   * Add planning runs every few ticks to choose better actions
6. Phase 6: Communication
   * Enable vocal actions
   * Add tiny reward for vocalizing near other agents
   * Let meaning emerge from reward correlations

Each phase builds on the previous and keeps the complexity under control.

---
