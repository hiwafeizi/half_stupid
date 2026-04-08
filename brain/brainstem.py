"""Brainstem — Survival reflex module. 2-hidden-layer neural network.

Purpose: eat when hungry, avoid danger, fight back when attacked.
This module gets FROZEN once trained. Higher modules (PFC etc.) build on top.

Zero hand-crafted features. No guidance. Raw numbers only.

Input layout (135 floats):
    [0-24]    Vision layer y=-2 (below feet — cliff detection)
    [25-49]   Vision layer y=-1 (floor — what you stand on)
    [50-74]   Vision layer y=0  (eye level — walls, food, mobs)
    [75-99]   Vision layer y=1  (above head — ceiling, falling blocks)
              5x5 grid per layer, 4 layers = 100 raw block IDs.
              Each block = integer assigned on first encounter per agent.

    [100]     Health (0-20)
    [101]     Food level (0-20)
    [102]     X position
    [103]     Y position
    [104]     Z position
    [105]     Yaw (0-360, facing direction)
    [106]     Pitch (-90 to 90, look up/down)

    [107-115] Hotbar item IDs (9 slots, -1 = empty)
    [116-124] Hotbar item counts (9 slots, 0-64)
    [125]     Held item ID (-1 if empty)
    [126]     Held item count (0-64)

    [127]     Active: eating/use (0 or 1)
    [128]     Active: moving (0 or 1)
    [129]     Active: strafing (0 or 1)
    [130]     Active: turning (0 or 1)
    [131]     Active: pitching (0 or 1)
    [132]     Active: jumping (0 or 1)
    [133]     Active: crouching (0 or 1)
    [134]     Active: attacking (0 or 1)

Input masking (reveal step by step):
    Level 1: health + food + eating flag (3 active)
    Level 2: + held item ID + count (5 active)
    Level 3: + hotbar IDs + counts (23 active)
    Level 4: + vision + movement action flags (128 active)
    Level 5: + x,y,z,yaw,pitch = everything (135 active)

Actions (23 total, keyboard-like):
    0:  W forward       8:  Space jump        13-21: hotbar 1-9
    1:  S backward      9:  Shift crouch      22: stand still
    2:  A strafe left   10: RClick use/eat
    3:  D strafe right  11: LClick attack
    4:  Turn left       12: Q throw
    5:  Turn right
    6:  Look up
    7:  Look down

Action masking (reveal step by step):
    Level 1: eat + still (2)
    Level 2: + hotbar 1-3 (5)
    Level 3: + hotbar 4-9 (11)
    Level 4: + walk + turn (15)
    Level 5: + strafe, jump, crouch, attack, throw (21)
    Level 6: everything (23)

Architecture:
    Input (135) -> Hidden 1 (64, ReLU) -> Hidden 2 (32, ReLU) -> Output (23, softmax)
    ~11,400 weights. Sized for survival reflexes only.

Learning:
    REINFORCE with per-timestep analytical gradients.
    Reward: +1 alive, large negative on death. Nothing else.

Per-agent saves (each agent has its own directory):
    weights.npz  — network weights (w1, b1, w2, b2, w3, b3)
    vocab.json   — block vocabulary, input/action levels
    history.json — full episode-by-episode stats
"""

import numpy as np
import json
from pathlib import Path


# ─── Actions (keyboard mapping) ───────────────────────────────

ACTIONS = [
    # Movement
    ("move", 1.0),               # 0:  W — forward
    ("move", -1.0),              # 1:  S — backward
    ("strafe", -1.0),            # 2:  A — strafe left
    ("strafe", 1.0),             # 3:  D — strafe right
    ("turn", -1.0),              # 4:  mouse left — turn left
    ("turn", 1.0),               # 5:  mouse right — turn right
    ("pitch", -1.0),             # 6:  mouse up — look up
    ("pitch", 1.0),              # 7:  mouse down — look down
    ("jump", 1),                 # 8:  space — jump
    ("crouch", 1),               # 9:  shift — crouch/sneak
    # Interaction
    ("use", 1),                  # 10: right click — use/eat/place
    ("attack", 1),               # 11: left click — attack/break
    ("discardCurrentItem", 1),   # 12: Q — throw/drop item
    # Hotbar selection
    ("hotbar.1", 1),             # 13: key 1
    ("hotbar.2", 1),             # 14: key 2
    ("hotbar.3", 1),             # 15: key 3
    ("hotbar.4", 1),             # 16: key 4
    ("hotbar.5", 1),             # 17: key 5
    ("hotbar.6", 1),             # 18: key 6
    ("hotbar.7", 1),             # 19: key 7
    ("hotbar.8", 1),             # 20: key 8
    ("hotbar.9", 1),             # 21: key 9
    # Nothing
    ("move", 0.0),               # 22: stand still
]
NUM_ACTIONS = len(ACTIONS)

# Vision: 5x5 grid, 4 height layers
GRID_W = 5
GRID_H = 4                  # y=-2 (below feet), y=-1 (floor), y=0 (eye), y=1 (above head)
GRID_SIZE = GRID_W * GRID_W * GRID_H  # 5*5*4 = 100 blocks

# Input layout (135 total):
# [0-24]    vision layer y=-2 (below feet — cliff detection)
# [25-49]   vision layer y=-1 (floor — what you stand on)
# [50-74]   vision layer y=0  (eye level — walls, food, mobs)
# [75-99]   vision layer y=1  (above head — ceiling, falling blocks)
# [100]     health
# [101]     food
# [102]     x
# [103]     y
# [104]     z
# [105]     yaw
# [106]     pitch
# [107-115] 9 hotbar item IDs
# [116-124] 9 hotbar item counts
# [125]     held item ID
# [126]     held item count
# [127]     active: eating/use (0 or 1)
# [128]     active: moving (0 or 1)
# [129]     active: strafing (0 or 1)
# [130]     active: turning (0 or 1)
# [131]     active: pitching (0 or 1)
# [132]     active: jumping (0 or 1)
# [133]     active: crouching (0 or 1)
# [134]     active: attacking (0 or 1)
INPUT_DIM = 135

# Network — sized for survival reflexes (eat, avoid danger, fight back)
HIDDEN1 = 64
HIDDEN2 = 32

# ─── Input masking ────────────────────────────────────────────
# Reveal inputs gradually across stages.
# Network shape stays the same — masked inputs are just 0.
# This lets weights trained in Stage 1 carry over to Stage 2+
# without reshaping the network.
#
# Mask is a dict: input_index → True (visible) or False (zeroed out)
# By default everything is visible. Set a stage mask to restrict.

# ─── Input mask levels ────────────────────────────────────────
# Reveal inputs step by step. Higher level = more inputs.
# Network shape stays 135 always. Masked inputs are zeroed.
# Weights carry over when you upgrade level.

MASK_LEVELS = {
    # Level 1: health + food + eating flag (3 active)
    # "Am I dying? Am I eating?"
    1: {
        "vision": False,
        "health": True,
        "food": True,
        "x": False, "y": False, "z": False,
        "yaw": False, "pitch": False,
        "hotbar_items": False,
        "held_item": False,
        "eating_flag": True,
        "other_action_flags": False,
    },
    # Level 2: + held item ID + count (5 active)
    # "What am I holding? How much left?"
    2: {
        "vision": False,
        "health": True,
        "food": True,
        "x": False, "y": False, "z": False,
        "yaw": False, "pitch": False,
        "hotbar_items": False,
        "held_item": True,
        "eating_flag": True,
        "other_action_flags": False,
    },
    # Level 3: + hotbar IDs + counts (23 active)
    # "What's in all my slots?"
    3: {
        "vision": False,
        "health": True,
        "food": True,
        "x": False, "y": False, "z": False,
        "yaw": False, "pitch": False,
        "hotbar_items": True,
        "held_item": True,
        "eating_flag": True,
        "other_action_flags": False,
    },
    # Level 4: + vision (100 blocks) + 7 movement action flags (130 active)
    # "What's around me? Am I moving?"
    4: {
        "vision": True,
        "health": True,
        "food": True,
        "x": False, "y": False, "z": False,
        "yaw": False, "pitch": False,
        "hotbar_items": True,
        "held_item": True,
        "eating_flag": True,
        "other_action_flags": True,
    },
    # Level 5: everything (135 active)
    # "Full awareness"
    5: {
        "vision": True,
        "health": True,
        "food": True,
        "x": True, "y": True, "z": True,
        "yaw": True, "pitch": True,
        "hotbar_items": True,
        "held_item": True,
        "eating_flag": True,
        "other_action_flags": True,
    },
}

# ─── Output (action) masking ─────────────────────────────────
# Same idea: fewer choices = learns faster.
# Masked actions get zero probability — agent can't pick them.
#
# Action indices:
#  0-3: move (W,S,A,D)    4-5: turn    6-7: pitch
#  8: jump    9: crouch    10: use/eat  11: attack
#  12: throw  13-21: hotbar 1-9         22: stand still

ACTION_MASK_LEVELS = {
    # Level 1: Only eat or do nothing. 2 actions.
    1: [10, 22],                          # use/eat, stand still

    # Level 2: + hotbar 1-3 only. 5 actions. Learn to switch between 3 slots.
    2: [10, 13, 14, 15, 22],              # use + hotbar 1-3 + still

    # Level 3: + hotbar 4-9. All slots. 11 actions.
    3: [10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],  # use + hotbar 1-9 + still

    # Level 4: + basic movement. 15 actions.
    4: [0, 1, 4, 5, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],  # + move fwd/back + turn

    # Level 5: + all movement + interaction. 21 actions.
    5: [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],

    # Level 6: Everything. All 23 actions.
    6: list(range(NUM_ACTIONS)),
}


def _build_action_mask(stage: int) -> np.ndarray:
    """Build binary mask (23,) for allowed actions at this stage."""
    active = 1
    for s in sorted(ACTION_MASK_LEVELS.keys()):
        if s <= stage:
            active = s

    allowed = ACTION_MASK_LEVELS[active]
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for idx in allowed:
        mask[idx] = 1.0
    return mask


def _build_mask(stage: int) -> np.ndarray:
    """Build a binary mask array (135,) for the given stage."""
    active_stage = 1
    for s in sorted(MASK_LEVELS.keys()):
        if s <= stage:
            active_stage = s

    cfg = MASK_LEVELS[active_stage]
    # Layout: [100 vision | 7 body | 9 hotbar IDs | 9 hotbar counts |
    #          held_id | held_count | eating | 7 action flags]
    #          0-99       100-106  107-115       116-124
    #          125        126      127      128-134
    mask = np.zeros(INPUT_DIM, dtype=np.float32)

    if cfg["vision"]:
        mask[0:100] = 1.0                   # 0-99   (5x5x4 vision)
    if cfg["health"]:
        mask[100] = 1.0                     # 100
    if cfg["food"]:
        mask[101] = 1.0                     # 101
    if cfg["x"]:
        mask[102] = 1.0                     # 102
    if cfg["y"]:
        mask[103] = 1.0                     # 103
    if cfg["z"]:
        mask[104] = 1.0                     # 104
    if cfg["yaw"]:
        mask[105] = 1.0                     # 105
    if cfg["pitch"]:
        mask[106] = 1.0                     # 106
    if cfg["hotbar_items"]:
        mask[107:116] = 1.0                 # 107-115 (9 IDs)
        mask[116:125] = 1.0                 # 116-124 (9 counts)
    if cfg["held_item"]:
        mask[125] = 1.0                     # 125 held item ID
        mask[126] = 1.0                     # 126 held item count
    if cfg["eating_flag"]:
        mask[127] = 1.0                     # 127 eating/use active
    if cfg["other_action_flags"]:
        mask[128:135] = 1.0                 # 128-134 other 7 action flags

    return mask


class Brainstem:
    """3-hidden-layer neural network brain with learned block embeddings.

    No hand-crafted features. Each agent has its own:
        - Block vocabulary (maps block names → integer IDs)
        - Embedding matrix (maps IDs → learned vectors)
        - Network weights
        - Training history
    """

    def __init__(self, name: str = "agent", learning_rate: float = 0.001,
                 input_level: int = 1, action_level: int = 1, stage: int = None,
                 min_exploration: float = 0.0005):
        self.name = name
        self.lr = learning_rate
        self.min_exploration = min_exploration
        # Support old 'stage' param for backwards compat
        if stage is not None:
            input_level = stage
            action_level = stage
        self.input_level = input_level
        self.action_level = action_level

        # Input mask — zeros out inputs not yet revealed
        self.mask = _build_mask(input_level)
        # Action mask — zeros out actions not yet available
        self.action_mask = _build_action_mask(action_level)

        # Block vocabulary — built up as agent encounters new blocks
        # Each agent independently assigns IDs: "object_0", "object_1", ...
        self.block_vocab = {}    # block_name → integer ID
        self.next_block_id = 0
        self.max_vocab = 256     # max unique block types

        # Network weights — Xavier initialization
        # Layer 1: input → hidden1
        self.w1 = np.random.randn(INPUT_DIM, HIDDEN1) * np.sqrt(2.0 / INPUT_DIM)
        self.b1 = np.zeros(HIDDEN1)
        # Layer 2: hidden1 → hidden2
        self.w2 = np.random.randn(HIDDEN1, HIDDEN2) * np.sqrt(2.0 / HIDDEN1)
        self.b2 = np.zeros(HIDDEN2)
        # Layer 3: hidden2 → actions
        self.w3 = np.random.randn(HIDDEN2, NUM_ACTIONS) * np.sqrt(2.0 / HIDDEN2)
        self.b3 = np.zeros(NUM_ACTIONS)

        # Per-timestep episode buffer for REINFORCE
        self._ep_inputs = []
        self._ep_h1s = []
        self._ep_h2s = []
        self._ep_probs = []
        self._ep_actions = []
        self._ep_rewards = []

        # Training history — full stats per episode
        self.episodes_trained = 0
        self.total_ticks = 0
        self.episode_history = []  # list of dicts, one per episode

    def _get_block_id(self, block_name: str) -> int:
        """Get or assign integer ID for a block type.

        Each agent builds its own vocabulary independently.
        First time seeing "stone"? It becomes object_0.
        First time seeing "cake"? It becomes object_1. Etc.
        """
        if block_name not in self.block_vocab:
            if self.next_block_id < self.max_vocab:
                self.block_vocab[block_name] = self.next_block_id
                self.next_block_id += 1
            else:
                return self.max_vocab - 1  # overflow bucket
        return self.block_vocab[block_name]

    def encode_vision(self, grid_blocks: list) -> np.ndarray:
        """Convert raw block names to integer ID vector.

        5x5 grid, 4 height layers = 100 blocks.
        Each block name → integer ID. No features, no embedding.

        Returns:
            np.ndarray shape (100,) of float block IDs.
        """
        block_ids = []
        for block in grid_blocks[:GRID_SIZE]:
            bid = self._get_block_id(str(block))
            block_ids.append(float(bid))

        # Pad if grid is smaller than expected
        while len(block_ids) < GRID_SIZE:
            block_ids.append(0.0)

        return np.array(block_ids[:GRID_SIZE], dtype=np.float32)

    def encode_internal(self, obs: dict, active_actions: dict = None) -> np.ndarray:
        """Extract raw internal state — no normalization, no feature engineering.

        Returns 35 floats:
            [health, food, x, y, z, yaw, pitch,          # 7 body
             hotbar_0_id, ..., hotbar_8_id,                # 9 hotbar IDs
             hotbar_0_count, ..., hotbar_8_count,          # 9 hotbar counts
             held_item_id, held_item_count,                # 2 current held item
             eating_flag,                                  # 1 am I eating?
             moving, strafing, turning, pitching,          # 7 other action flags
             jumping, crouching, attacking]

        active_actions: dict of currently active continuous commands
        """
        state = [
            obs.get("Life", 0.0),
            obs.get("Food", 0.0),
            obs.get("XPos", 0.0),
            obs.get("YPos", 0.0),
            obs.get("ZPos", 0.0),
            obs.get("Yaw", 0.0),
            obs.get("Pitch", 0.0),
        ]

        # Hotbar item IDs + counts: 9 slots each
        counts = []
        for slot in range(9):
            count = int(obs.get(f"Hotbar_{slot}_size", 0))
            counts.append(count)
            item_name = obs.get(f"Hotbar_{slot}_item", "")
            if item_name and count > 0:
                state.append(float(self._get_block_id(str(item_name))))
            else:
                state.append(-1.0)

        for count in counts:
            state.append(float(count))

        # Held item: ID + count of currently selected slot
        current_idx = int(obs.get("currentItemIndex", 0))
        current_item_name = obs.get(f"Hotbar_{current_idx}_item", "")
        current_item_count = int(obs.get(f"Hotbar_{current_idx}_size", 0))
        if current_item_name and current_item_count > 0:
            state.append(float(self._get_block_id(str(current_item_name))))
        else:
            state.append(-1.0)
        state.append(float(current_item_count))

        # Active action flags (8 continuous commands)
        if active_actions is None:
            active_actions = {}
        state.append(float(active_actions.get("use", 0)))       # 125: eating
        state.append(float(active_actions.get("move", 0)))      # 126: moving
        state.append(float(active_actions.get("strafe", 0)))    # 127: strafing
        state.append(float(active_actions.get("turn", 0)))      # 128: turning
        state.append(float(active_actions.get("pitch", 0)))     # 129: pitching
        state.append(float(active_actions.get("jump", 0)))      # 130: jumping
        state.append(float(active_actions.get("crouch", 0)))    # 131: crouching
        state.append(float(active_actions.get("attack", 0)))    # 132: attacking

        return np.array(state, dtype=np.float32)

    def forward(self, vision: np.ndarray, internal: np.ndarray):
        """Forward pass: input → h1 (ReLU) → h2 (ReLU) → output (softmax).

        Applies stage mask — zeroes out inputs not yet revealed.
        Returns: (probs, input_vec, h1, h2)
        """
        x = np.concatenate([vision, internal])
        x = x * self.mask  # zero out masked inputs

        # Layer 1
        h1 = x @ self.w1 + self.b1
        h1 = np.maximum(0, h1)  # ReLU

        # Layer 2
        h2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0, h2)  # ReLU

        # Output layer + action mask + softmax + minimum exploration
        logits = h2 @ self.w3 + self.b3
        # Mask: set disabled actions to -inf so softmax gives them 0
        logits = np.where(self.action_mask > 0, logits, -1e9)
        logits -= np.max(logits)
        exp_l = np.exp(logits)
        probs = exp_l / (exp_l.sum() + 1e-8)

        # Minimum probability for enabled actions (prevents dead exploration)
        min_prob = self.min_exploration
        num_enabled = self.action_mask.sum()
        if num_enabled > 0:
            probs = np.where(self.action_mask > 0,
                             np.maximum(probs, min_prob), 0.0)
            probs = probs / (probs.sum() + 1e-8)  # renormalize

        return probs, x, h1, h2

    def choose_action(self, obs: dict, grid_blocks: list, active_actions: dict = None) -> int:
        """Full pipeline: observe → encode → forward → sample action.

        Stores everything needed for learning.
        Returns: action index (0-22).
        """
        vision = self.encode_vision(grid_blocks)
        internal = self.encode_internal(obs, active_actions)
        probs, x, h1, h2 = self.forward(vision, internal)

        # Sample action
        action = np.random.choice(NUM_ACTIONS, p=probs)

        # Save for REINFORCE
        self._ep_inputs.append(x)
        self._ep_h1s.append(h1)
        self._ep_h2s.append(h2)
        self._ep_probs.append(probs.copy())
        self._ep_actions.append(action)

        self.total_ticks += 1
        return action

    def record_reward(self, reward: float):
        """Record reward for the current timestep."""
        self._ep_rewards.append(reward)

    def update(self, gamma: float = 0.99) -> float:
        """REINFORCE update at end of episode.

        Computes per-timestep policy gradients and updates:
            - w1, b1 (layer 1)
            - w2, b2 (layer 2)
            - w3, b3 (output layer)
            - embeddings (block embedding matrix)

        Returns: average reward this episode.
        """
        if not self._ep_rewards:
            return 0.0

        T = min(len(self._ep_rewards), len(self._ep_actions))
        avg_reward = np.mean(self._ep_rewards[:T])

        # Discounted returns
        returns = np.zeros(T, dtype=np.float32)
        R = 0.0
        for t in reversed(range(T)):
            R = self._ep_rewards[t] + gamma * R
            returns[t] = R

        # Normalize
        if T > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Accumulate gradients
        dw1 = np.zeros_like(self.w1)
        db1 = np.zeros_like(self.b1)
        dw2 = np.zeros_like(self.w2)
        db2 = np.zeros_like(self.b2)
        dw3 = np.zeros_like(self.w3)
        db3 = np.zeros_like(self.b3)

        for t in range(T):
            x = self._ep_inputs[t]
            h1 = self._ep_h1s[t]
            h2 = self._ep_h2s[t]
            probs = self._ep_probs[t]
            action = self._ep_actions[t]
            G = returns[t]

            # Output gradient: d_logits = (one_hot - probs) * G
            d_logits = -probs.copy()
            d_logits[action] += 1.0
            d_logits *= G

            # Layer 3 gradients
            dw3 += np.outer(h2, d_logits)
            db3 += d_logits

            # Backprop to layer 2
            d_h2 = d_logits @ self.w3.T
            d_h2 *= (h2 > 0).astype(np.float32)

            dw2 += np.outer(h1, d_h2)
            db2 += d_h2

            # Backprop to layer 1
            d_h1 = d_h2 @ self.w2.T
            d_h1 *= (h1 > 0).astype(np.float32)

            dw1 += np.outer(x, d_h1)
            db1 += d_h1

        # Apply gradients
        scale = self.lr / max(T, 1)
        self.w3 += scale * dw3
        self.b3 += scale * db3
        self.w2 += scale * dw2
        self.b2 += scale * db2
        self.w1 += scale * dw1
        self.b1 += scale * db1

        # Record history
        self.episodes_trained += 1
        # survival/deaths/food recorded externally via record_episode_stats()

        # Clear episode buffer
        self._ep_inputs.clear()
        self._ep_h1s.clear()
        self._ep_h2s.clear()
        self._ep_probs.clear()
        self._ep_actions.clear()
        self._ep_rewards.clear()

        return avg_reward

    def save(self, directory: str):
        """Save everything for this agent to a directory.

        Saves:
            - weights.npz (network weights + embeddings)
            - vocab.json (block vocabulary)
            - history.json (training stats)
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        # Weights + embeddings
        np.savez(str(d / "weights.npz"),
                 w1=self.w1, b1=self.b1,
                 w2=self.w2, b2=self.b2,
                 w3=self.w3, b3=self.b3)

        # Vocabulary + stage
        with open(d / "vocab.json", "w") as f:
            json.dump({
                "block_vocab": self.block_vocab,
                "next_block_id": self.next_block_id,
                "input_level": self.input_level,
                "action_level": self.action_level,
            }, f, indent=2)

        # Training history
        with open(d / "history.json", "w") as f:
            json.dump({
                "name": self.name,
                "episodes_trained": self.episodes_trained,
                "total_ticks": self.total_ticks,
                "episode_history": self.episode_history,
                "vocab_size": self.next_block_id,
            }, f, indent=2)

    def load(self, directory: str):
        """Load saved agent state from directory."""
        d = Path(directory)

        # Weights
        data = np.load(str(d / "weights.npz"))
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
        self.w3 = data["w3"]
        self.b3 = data["b3"]

        # Vocabulary
        with open(d / "vocab.json") as f:
            vocab_data = json.load(f)
            self.block_vocab = vocab_data["block_vocab"]
            self.next_block_id = vocab_data["next_block_id"]
            if "input_level" in vocab_data:
                self.input_level = vocab_data["input_level"]
                self.action_level = vocab_data["action_level"]
                self.mask = _build_mask(self.input_level)
                self.action_mask = _build_action_mask(self.action_level)
            elif "stage" in vocab_data:
                self.set_stage(vocab_data["stage"])

        # History
        if (d / "history.json").exists():
            with open(d / "history.json") as f:
                hist = json.load(f)
                self.episodes_trained = hist.get("episodes_trained", 0)
                self.total_ticks = hist.get("total_ticks", 0)
                self.episode_history = hist.get("episode_history", [])

    def record_episode_stats(self, episode_data: dict):
        """Record full episode stats (called from training loop)."""
        self.episode_history.append(episode_data)

    def set_levels(self, input_level: int, action_level: int):
        """Change input/action mask levels independently.

        Network weights stay the same. New inputs/actions just unlock.
        """
        self.input_level = input_level
        self.action_level = action_level
        self.mask = _build_mask(input_level)
        self.action_mask = _build_action_mask(action_level)

    def set_stage(self, stage: int):
        """Set both input and action level to the same value."""
        self.set_levels(stage, stage)

    def __repr__(self):
        return (f"Brainstem('{self.name}', input={INPUT_DIM}, "
                f"h1={HIDDEN1}, h2={HIDDEN2}, actions={NUM_ACTIONS}, "
                f"vocab={self.next_block_id}, episodes={self.episodes_trained})")
