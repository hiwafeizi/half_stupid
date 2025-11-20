## Practical to do list

This is a concrete checklist you can literally paste into a project TODO.

### Stage 0: Project setup

* [ ] Decide platform
  * [ ] Minecraft + Malmo or custom grid world
* [ ] Create repo and base structure
* [ ] Implement basic environment API
  * [ ] Step function
  * [ ] State representation
  * [ ] Reward return

### Stage 1: Perception and memory skeleton

* [ ] Implement `sense_environment` function
  * [ ] Vision patch around agent
  * [ ] Basic sound events interface
  * [ ] Internal state extraction
* [ ] Implement STM data structure
  * [ ] Append recent events with time and reward
  * [ ] Truncate history to max length
* [ ] Implement LTM skeleton
  * [ ] Dictionaries for action_stats, vocal_meanings, danger_tags
  * [ ] Functions to update running averages
* [ ] Implement SPATIAL memory
  * [ ] Coordinate based dictionary
  * [ ] Reward accumulation per tile

### Stage 2: Reflex module

* [ ] Design reflex input vector (local tiles + sound + fall + health)
* [ ] Implement tiny neural network for reflex (or placeholder rules)
* [ ] Add priority logic so reflex action overrides others
* [ ] Test in simple environment where agent must avoid lava or cliffs

### Stage 3: Fast PFC

* [ ] Design input vector for fast PFC (compressed perception + simple STM)
* [ ] Implement heuristic or small network that:
  * [ ] Moves toward visible food
  * [ ] Avoids visible danger
  * [ ] Explores unknown tiles
* [ ] Integrate with reflex module in tick loop

### Stage 4: RL policy (reflective PFC)

* [ ] Choose RL library or implement PPO / A2C
* [ ] Define policy network architecture
* [ ] Connect environment, STM, and LTM summaries into state vector
* [ ] Implement training loop
* [ ] Test in simple environment until policy improves over heuristics

### Stage 5: Memory value tagging

* [ ] On each step, log: state hash, action, reward into LTM
* [ ] Implement update of action_stats (avg_reward, success_rate)
* [ ] Implement update of SPATIAL avg_reward per tile
* [ ] Use LTM summaries as extra features for reflective PFC and planning later

### Stage 6: Planning PFC

* [ ] Collect dataset of (state, action, next_state, reward) during RL episodes
* [ ] Train world model network to predict next_state and reward
* [ ] Implement planning routine that:
  * [ ] Simulates a few steps ahead for candidate actions
  * [ ] Uses SPATIAL and LTM risk statistics
  * [ ] Selects best action
* [ ] Call planner every N ticks and override lower level action when confident

### Stage 7: Communication

* [ ] Decide mapping of 5 letters to actual Minecraft sounds or note block pitches
* [ ] Implement vocal actions in the action space
* [ ] Implement parsing sound events into symbolic letters
* [ ] Update LTM for vocal_meanings based on reward co-occurrence
* [ ] Add small reward when vocalizing near another agent
* [ ] Evaluate if simple meanings emerge (A = danger, B = food etc.)

### Stage 8: Tuning and polish

* [ ] Tune reward weights for survival, food, pain, social, and novelty
* [ ] Adjust reflex threshold smoothness
* [ ] Adjust how often planner runs to balance compute vs intelligence
* [ ] Add logging and visualization for:
  * [ ] Action stats
  * [ ] Spatial reward heatmap
  * [ ] Vocal meaning evolution
