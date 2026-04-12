## Practical to do list

Updated 2026-04-08. Following bottom-up 9-stage training curriculum.

### Stage 0: Project setup

* [X] Decide platform: Minecraft + Malmo (1.11.2)
* [X] Create repo and base structure
* [X] Implement 4-level vision pipeline (reflex, fast PFC, reflective PFC, planning PFC)
* [X] Set up Malmo integration (builder, multi-agent support)
* [X] Define 9-stage training curriculum
* [X] Core settings: 4 agents, 5x5x4 vision, configurable speed, 1 hidden layer NN

### Stage 1: Food in reach — learn to eat (Brainstem) ← CURRENT

**Architecture (PyTorch GPU):**
* [X] Embedding: 2048 vocab x 4 dims, shared across all block/item IDs
* [X] Network: 465 -> 32 (ReLU) -> 23 (softmax), ~16.6k params
* [X] 110 ID inputs (100 vision + 9 hotbar + 1 held) -> embedded to 440 floats
* [X] 25 raw float inputs (body, counts, action flags)
* [X] PyTorch + CUDA on RTX 4060 GPU
* [X] Adam optimizer, autograd handles all gradients
* [X] REINFORCE with configurable gamma (0.97)

**Inputs & actions:**
* [X] Vision: 5x5 grid x 4 height layers (y=-2 to y=1) = 100 block IDs
* [X] Zero features: raw IDs + raw floats, agent learns everything
* [X] Input masking: 5 levels (body+eating -> held -> hotbar -> vision+flags -> full)
* [X] Action masking: 6 levels (eat -> hotbar 1-3 -> hotbar 1-9 -> walk -> combat -> full)
* [X] Active action flags: 8 continuous command states as inputs
* [X] Held item ID + count as inputs

**Training infrastructure:**
* [X] Per-agent saves: weights.pt, vocab.json, history.json
* [X] Auto-load checkpoints on restart
* [X] Best model saved per agent (highest alive_seconds)
* [X] One life per episode (no respawn)
* [X] Hunger via /effect (food drains, starvation kills on hard)
* [X] Configurable: learning rate, gamma, game speed, min exploration
* [X] Randomized hotbar each episode (random foods, counts, slots, uneatable items)
* [X] /replaceitem for exact slot placement
* [X] /gamerule sendCommandFeedback false (no chat spam)

**Monitoring:**
* [X] Live debug JSON per agent (live_Adam.json) — shows exact AI state
* [X] Probs JSON per agent (probs_Adam.json) — action probabilities every 20 ticks
* [X] Full episode stats in history.json (survival%, seconds, deaths, food, reward)

**Training progress:**
* [X] Level 1 (eat + still): agents learned to eat, hold right-click for duration
* [X] Level 2 (+ held item): agents see what they're holding
* [X] Level 3 (+ hotbar): agents learning to switch slots, eat 15-27 items
* [ ] Generalization: verify with randomized food, uneatable items, random slots
* [ ] Graduate: 5 consecutive episodes eating all available food
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
