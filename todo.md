## Practical to do list

Updated 2026-04-04. Following bottom-up 9-stage training curriculum.

### Stage 0: Project setup

* [X] Decide platform: Minecraft + Malmo (1.11.2)
* [X] Create repo and base structure
* [X] Implement 4-level vision pipeline (reflex, fast PFC, reflective PFC, planning PFC)
* [X] Set up Malmo integration (builder, multi-agent support)
* [X] Define 9-stage training curriculum
* [X] Core settings: 4 agents, 7×7 vision, 20x speed, respawn, 2-layer NN

### Stage 1: Food in reach — learn to eat (Brainstem) ← CURRENT

* [X] Create Stage 1 world (16×16 grassy yard, daytime, fence walls, food nearby)
* [X] Build brainstem NN: 135 inputs → 64 hidden → 32 hidden → 23 actions (~11.4k weights)
* [X] Vision: 5×5 grid × 4 height layers (y=-2 to y=1) = 100 raw block IDs
* [X] Zero features: raw block IDs, raw internals, agent learns everything
* [X] Per-agent saves: weights, vocabulary, full episode history
* [X] Input masking: 5 levels (body+eating → held item → hotbar → vision+flags → full)
* [X] Action masking: 6 levels (eat → hotbar 1-3 → hotbar 1-9 → walk → combat → full)
* [X] Active action flags: 8 continuous command states as inputs
* [X] Held item ID + count as inputs
* [X] Item counts in hotbar inputs (9 IDs + 9 counts)
* [X] /give and /effect use agent names (not @p)
* [X] One life per episode (no respawn)
* [X] Hunger via /effect (food drains, starvation kills on hard)
* [X] Live debug JSON per agent (live_Adam.json etc.)
* [X] Full episode stats in history.json (survival%, seconds, deaths, food, reward)
* [X] Auto-load checkpoints on restart
* [X] Learning rate configurable (currently 0.01)
* [X] Best model saved per agent (highest alive_seconds)
* [X] Probs-only JSON log for monitoring action probabilities
* [ ] Verify survival improves across episodes
* [ ] Graduate to input/action level 2, then 3, etc.
* [ ] Generalization tests before moving on:
  * [ ] Randomized food order and slot positions each episode
  * [ ] Random item counts (1-5 per slot)
  * [ ] Some slots empty (4-9 filled out of 9)
  * [ ] Uneatable items in some slots (stone, stick) — agent must skip them
  * [ ] Verify agents still survive well with randomized setup
* [ ] Save Stage 1 "graduated" weights

### Stage 2: Food nearby — learn to find it (Brainstem+)

* [ ] Expand world: food exists but not immediately visible
* [ ] Add exploration drive (curiosity bonus for new tiles)
* [ ] Verify agents learn biased random walk toward food
* [ ] Save Stage 2 weights

### Stage 3: Food disappears — remember where it was (Hippocampus)

* [ ] Add respawning food at fixed locations
* [ ] Implement short-term spatial memory
* [ ] Verify agents learn to route to known food spots
* [ ] Save Stage 3 weights

### Stage 4: Hazards — learn to avoid (Amygdala)

* [ ] Add lava/damage zones
* [ ] Implement fear/aversion signal (negative reward on health loss)
* [ ] Verify agents learn to avoid correlated dangers
* [ ] Save Stage 4 weights

### Stage 5: Larger world — build a map (Hippocampus+)

* [ ] Expand arena, make food sparse
* [ ] Implement long-term spatial memory (persistent map)
* [ ] Verify persistent path planning emerges
* [ ] Save Stage 5 weights

### Stage 6: Food requires steps — learn to plan (Prefrontal)

* [ ] Require crafting/farming for food
* [ ] Implement sequential planning + delay tolerance
* [ ] Verify multi-step goal pursuit
* [ ] Save Stage 6 weights

### Stage 7: Other agents — model them (Social cortex)

* [ ] Multiple agents share same food (already 4 agents!)
* [ ] Implement agent detection + behavior tracking
* [ ] Verify agents treat others as entities, not obstacles
* [ ] Save Stage 7 weights

### Stage 8: Info sharing — primitive signals (Language proto)

* [ ] Enable vocal actions (5 symbols: A-E)
* [ ] Reward for communication that improves survival
* [ ] Verify emergent signals for food/danger
* [ ] Save Stage 8 weights

### Stage 9: Cooperation vs competition (Theory of mind)

* [ ] Mix agent types (cooperative/competitive)
* [ ] Implement belief model about other agents
* [ ] Verify trust, reputation, social strategy emerges
* [ ] Save Stage 9 weights
