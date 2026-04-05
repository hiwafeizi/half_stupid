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

* [X] Create Stage 1 world (10×10 room, food adjacent to spawns)
* [X] Build 2-layer brainstem NN (102 inputs → 64 hidden → 6 actions)
* [X] Create training runner with episode loop
* [X] 4 agents, single respawn point, 20x speed
* [ ] Launch 4 Malmo clients and run first training
* [ ] Verify agents can observe 7×7 grid
* [ ] Verify agents learn to eat when food is adjacent
* [ ] Tune reward/learning rate until survival improves across episodes
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
