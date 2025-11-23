# Vision Pipeline Notes (Draft)

## 1. Raw Malmo grid observations
- Every observation arrives as a flat list of block-id strings.
- Ordering: iterate Y (vertical) layers, then Z rows, then X columns.
- For floor slices we typically have a single Y layer, so reshape into `size × size` around the agent.

## 2. Layer sizes under consideration
- **Reflex system**: keep a tight 3×3 footprint for immediate safety checks.
- **Fast PFC**: candidate upgrade from 5×5 to 7×7 so quick heuristics see approaching hazards.
- **Reflective PFC**: increase to either 11×11 or 13×13. Goal is to cover near-future pathing while still feeding a compact CNN/MLP.
- **Planning PFC**: grow to **31×31** (captures ~15 blocks in every direction). Acts as the world-model inlet.

## 3. Compression strategy for larger fields
- Do not pass raw 31×31×feature tensors each tick.
- Split data into **multi-resolution summaries**:
  1. Preserve a high-detail core (eg. center 9×9) for precise navigation.
  2. Downsample outer rings using average pooling or learned embeddings (eg. 3×3 pooling -> 10×10 coarse grid).
  3. Maintain running statistics (max danger, min walkability) per coarse cell for quick threshold checks.
- For categorical data (block ids), map to semantic features (walkable, danger, reward, friction, liquid, unknown). Store as floats to enable pooling.

## 4. Memory-backed overlays
- Planning layer should read/write heatmaps:
  - Risk accumulation
  - Reward sightings
  - Agent density / social pressure
  - Safe haven frequency
- Store overlays at the same 31×31 resolution but compress using:
  - Exponential decay per tick (limits historical weight)
  - Sparse encoding (dictionary keyed by coordinates) with periodic rasterization when needed

## 5. Incremental data storage
- Cache previous observation tensor and only transmit deltas (changed cells) each tick.
- Use XOR-like masks for boolean features (`is_unknown`, `is_liquid`).
- For float channels, quantize to 8-bit or 12-bit fixed point before persisting to the planner buffer.

## 6. Next questions to resolve
1. Exact pooling scheme for reflective (11×11 vs 13×13) and planning (31×31 -> 11×11 coarse?).
2. Whether planning should run every tick or staggered (eg. every 5 ticks) to amortise cost.
3. API design: what format does each vision layer expose (dict with tensors + summaries?).
4. How to align observation grids with world coordinates for long-lived memory entries.

---
*Keep this document as the living design space before implementing the next iteration of vision modules.*
