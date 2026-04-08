"""Brainstem — Survival reflex module with learned embeddings.

Purpose: eat when hungry, avoid danger, fight back when attacked.
This module gets FROZEN once trained. Higher modules build on top.

Zero hand-crafted features. No guidance. Raw numbers only.

Architecture:
    Embedding table: 2048 vocab × 4 dims = 8,192 params
    All block/item IDs go through the SAME shared embedding.

    ID inputs (110 total, each → 4 floats via embedding):
        [0-99]    100 vision blocks (5×5×4 grid)
        [100-108] 9 hotbar item IDs
        [109]     1 held item ID
        → 110 × 4 = 440 embedded floats

    Raw float inputs (25 total, pass through directly):
        health, food                    = 2
        x, y, z, yaw, pitch            = 5
        9 hotbar counts                 = 9
        held item count                 = 1
        8 active action flags           = 8

    Network: 465 → 32 (ReLU) → 23 (softmax)
    Total params: ~16,640

    Input masking: masked IDs get zero vector, masked floats get zero.

Input masking (reveal step by step):
    Level 1: health + food + eating flag (3 raw)
    Level 2: + held item ID embed + count (3 raw + 4 embed)
    Level 3: + hotbar IDs embed + counts (12 raw + 40 embed)
    Level 4: + vision embeds + movement flags (19 raw + 440 embed)
    Level 5: + x,y,z,yaw,pitch = everything (25 raw + 440 embed)

Actions (23 total, keyboard-like):
    0-3: movement (W,S,A,D)    4-5: turn    6-7: look up/down
    8: jump    9: crouch    10: use/eat    11: attack
    12: throw  13-21: hotbar 1-9           22: stand still

Action masking (reveal step by step):
    Level 1: eat + still (2)
    Level 2: + hotbar 1-3 (5)
    Level 3: + hotbar 4-9 (11)
    Level 4: + walk + turn (15)
    Level 5: + strafe, jump, crouch, attack, throw (21)
    Level 6: everything (23)

Learning:
    REINFORCE with per-timestep analytical gradients.
    Embedding gradients flow back and update the shared table.
    Reward: +1 alive, large negative on death. Nothing else.

Per-agent saves:
    weights.npz  — w1, b1, w2, b2, embeddings
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

ACTION_NAMES = [
    "W_forward", "S_backward", "A_strafe_left", "D_strafe_right",
    "turn_left", "turn_right", "look_up", "look_down",
    "jump", "crouch", "use_eat", "attack",
    "Q_throw", "hotbar_1", "hotbar_2", "hotbar_3", "hotbar_4",
    "hotbar_5", "hotbar_6", "hotbar_7", "hotbar_8", "hotbar_9",
    "stand_still"
]

# ─── Vision ──────────────────────────────────────────────────
GRID_W = 5
GRID_H = 4                     # y=-2, y=-1, y=0, y=1
GRID_SIZE = GRID_W * GRID_W * GRID_H  # 100

# ─── Embedding ───────────────────────────────────────────────
EMBED_DIM = 4                  # each ID → 4 learned floats
MAX_VOCAB = 2048               # max unique block/item types
EMPTY_ID = 0                   # ID for empty/nothing (-1 maps here)

# ─── Input layout ────────────────────────────────────────────
# 110 ID inputs → embedding → 440 floats
# 25 raw float inputs
# Total network input: 465
NUM_ID_INPUTS = 100 + 9 + 1   # 100 vision + 9 hotbar + 1 held = 110
NUM_RAW_INPUTS = 25            # 2 body + 5 position + 9 counts + 1 held_count + 8 flags
NETWORK_INPUT_DIM = NUM_ID_INPUTS * EMBED_DIM + NUM_RAW_INPUTS  # 440 + 25 = 465

# ─── Network ─────────────────────────────────────────────────
HIDDEN = 32                    # single hidden layer, 32 nodes


# ─── Input mask levels ───────────────────────────────────────
# Controls which inputs are visible at each level.
# Masked IDs get zero embedding, masked raw floats get zero.
#
# ID input indices (0-109):
#   [0-99]    vision blocks
#   [100-108] hotbar item IDs
#   [109]     held item ID
#
# Raw input indices (0-24):
#   [0]  health    [1]  food
#   [2]  x         [3]  y         [4]  z
#   [5]  yaw       [6]  pitch
#   [7-15]  hotbar counts (9)
#   [16] held item count
#   [17] eating flag
#   [18] moving    [19] strafing  [20] turning  [21] pitching
#   [22] jumping   [23] crouching [24] attacking

MASK_LEVELS = {
    # Level 1: health + food + eating flag (3 raw, 0 embed)
    1: {
        "vision": False,
        "hotbar_ids": False,
        "held_id": False,
        "health": True, "food": True,
        "position": False,
        "hotbar_counts": False,
        "held_count": False,
        "eating_flag": True,
        "other_flags": False,
    },
    # Level 2: + held item ID + count (4 raw, 4 embed)
    2: {
        "vision": False,
        "hotbar_ids": False,
        "held_id": True,
        "health": True, "food": True,
        "position": False,
        "hotbar_counts": False,
        "held_count": True,
        "eating_flag": True,
        "other_flags": False,
    },
    # Level 3: + hotbar IDs + counts (13 raw, 40 embed)
    3: {
        "vision": False,
        "hotbar_ids": True,
        "held_id": True,
        "health": True, "food": True,
        "position": False,
        "hotbar_counts": True,
        "held_count": True,
        "eating_flag": True,
        "other_flags": False,
    },
    # Level 4: + vision + movement flags (20 raw, 440 embed)
    4: {
        "vision": True,
        "hotbar_ids": True,
        "held_id": True,
        "health": True, "food": True,
        "position": False,
        "hotbar_counts": True,
        "held_count": True,
        "eating_flag": True,
        "other_flags": True,
    },
    # Level 5: everything (25 raw, 440 embed)
    5: {
        "vision": True,
        "hotbar_ids": True,
        "held_id": True,
        "health": True, "food": True,
        "position": True,
        "hotbar_counts": True,
        "held_count": True,
        "eating_flag": True,
        "other_flags": True,
    },
}

# ─── Action mask levels ──────────────────────────────────────
ACTION_MASK_LEVELS = {
    1: [10, 22],                          # eat + still
    2: [10, 13, 14, 15, 22],              # + hotbar 1-3
    3: [10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],  # + hotbar 4-9
    4: [0, 1, 4, 5, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],  # + move + turn
    5: [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    6: list(range(NUM_ACTIONS)),           # everything
}


def _build_masks(input_level: int, action_level: int):
    """Build ID mask (110,), raw mask (25,), and action mask (23,)."""
    # Find effective levels
    il = max(l for l in MASK_LEVELS if l <= input_level)
    al = max(l for l in ACTION_MASK_LEVELS if l <= action_level)

    cfg = MASK_LEVELS[il]

    # ID mask (110): which IDs get embedded vs zeroed
    id_mask = np.zeros(NUM_ID_INPUTS, dtype=np.float32)
    if cfg["vision"]:
        id_mask[0:100] = 1.0           # vision blocks
    if cfg["hotbar_ids"]:
        id_mask[100:109] = 1.0         # hotbar IDs
    if cfg["held_id"]:
        id_mask[109] = 1.0             # held item ID

    # Raw mask (25): which raw floats pass through
    raw_mask = np.zeros(NUM_RAW_INPUTS, dtype=np.float32)
    if cfg["health"]:
        raw_mask[0] = 1.0              # health
    if cfg["food"]:
        raw_mask[1] = 1.0              # food
    if cfg["position"]:
        raw_mask[2:7] = 1.0            # x,y,z,yaw,pitch
    if cfg["hotbar_counts"]:
        raw_mask[7:16] = 1.0           # 9 hotbar counts
    if cfg["held_count"]:
        raw_mask[16] = 1.0             # held count
    if cfg["eating_flag"]:
        raw_mask[17] = 1.0             # eating
    if cfg["other_flags"]:
        raw_mask[18:25] = 1.0          # 7 other action flags

    # Action mask
    allowed = ACTION_MASK_LEVELS[al]
    action_mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for idx in allowed:
        action_mask[idx] = 1.0

    return id_mask, raw_mask, action_mask


class Brainstem:
    """Survival reflex network with learned block/item embeddings.

    No hand-crafted features. Each agent has its own:
        - Block vocabulary (maps names → integer IDs)
        - Embedding table (maps IDs → 4-dim learned vectors)
        - Network weights (465 → 32 → 23)
        - Training history
    """

    def __init__(self, name: str = "agent", learning_rate: float = 0.001,
                 input_level: int = 1, action_level: int = 1,
                 min_exploration: float = 0.0002):
        self.name = name
        self.lr = learning_rate
        self.min_exploration = min_exploration
        self.input_level = input_level
        self.action_level = action_level

        # Build masks
        self.id_mask, self.raw_mask, self.action_mask = \
            _build_masks(input_level, action_level)

        # Block vocabulary — built per agent on first encounter
        self.block_vocab = {}
        self.next_block_id = 1       # 0 reserved for EMPTY
        self.block_vocab["EMPTY"] = EMPTY_ID

        # Embedding table: shared across all ID inputs
        # Xavier init scaled for embed dim
        self.embeddings = np.random.randn(MAX_VOCAB, EMBED_DIM) * np.sqrt(2.0 / EMBED_DIM)
        # EMPTY embedding starts at zero
        self.embeddings[EMPTY_ID] = 0.0

        # Network: 465 → 32 → 23
        self.w1 = np.random.randn(NETWORK_INPUT_DIM, HIDDEN) * np.sqrt(2.0 / NETWORK_INPUT_DIM)
        self.b1 = np.zeros(HIDDEN)
        self.w2 = np.random.randn(HIDDEN, NUM_ACTIONS) * np.sqrt(2.0 / HIDDEN)
        self.b2 = np.zeros(NUM_ACTIONS)

        # Episode buffer for REINFORCE
        self._ep_inputs = []       # full 465-dim vectors
        self._ep_id_indices = []   # raw ID indices (110,) for embedding gradient
        self._ep_h1s = []
        self._ep_probs = []
        self._ep_actions = []
        self._ep_rewards = []

        # Training history
        self.episodes_trained = 0
        self.total_ticks = 0
        self.episode_history = []

    def _get_block_id(self, block_name: str) -> int:
        """Get or assign integer ID for a block/item type."""
        if not block_name or block_name == "air":
            return EMPTY_ID
        if block_name not in self.block_vocab:
            if self.next_block_id < MAX_VOCAB:
                self.block_vocab[block_name] = self.next_block_id
                self.next_block_id += 1
            else:
                return MAX_VOCAB - 1  # overflow bucket
        return self.block_vocab[block_name]

    def _encode(self, obs: dict, grid_blocks: list, active_actions: dict = None):
        """Encode observation into ID array (110,) and raw array (25,).

        Returns: (id_indices, raw_floats)
        """
        if active_actions is None:
            active_actions = {}

        # === ID inputs (110) ===
        ids = np.zeros(NUM_ID_INPUTS, dtype=np.int32)

        # Vision: 100 blocks
        for i, block in enumerate(grid_blocks[:GRID_SIZE]):
            ids[i] = self._get_block_id(str(block))
        # Pad if short
        # (already zeros = EMPTY_ID)

        # Hotbar item IDs: 9 slots
        for slot in range(9):
            item_name = obs.get(f"Hotbar_{slot}_item", "")
            count = int(obs.get(f"Hotbar_{slot}_size", 0))
            if item_name and count > 0:
                ids[100 + slot] = self._get_block_id(str(item_name))
            # else stays 0 = EMPTY_ID

        # Held item ID
        current_idx = int(obs.get("currentItemIndex", 0))
        held_name = obs.get(f"Hotbar_{current_idx}_item", "")
        held_count = int(obs.get(f"Hotbar_{current_idx}_size", 0))
        if held_name and held_count > 0:
            ids[109] = self._get_block_id(str(held_name))

        # === Raw inputs (25) ===
        raw = np.zeros(NUM_RAW_INPUTS, dtype=np.float32)
        raw[0] = obs.get("Life", 0.0)          # health
        raw[1] = obs.get("Food", 0.0)          # food
        raw[2] = obs.get("XPos", 0.0)          # x
        raw[3] = obs.get("YPos", 0.0)          # y
        raw[4] = obs.get("ZPos", 0.0)          # z
        raw[5] = obs.get("Yaw", 0.0)           # yaw
        raw[6] = obs.get("Pitch", 0.0)         # pitch

        # Hotbar counts
        for slot in range(9):
            raw[7 + slot] = float(obs.get(f"Hotbar_{slot}_size", 0))

        # Held item count
        raw[16] = float(held_count)

        # Active action flags
        raw[17] = float(active_actions.get("use", 0))       # eating
        raw[18] = float(active_actions.get("move", 0))      # moving
        raw[19] = float(active_actions.get("strafe", 0))    # strafing
        raw[20] = float(active_actions.get("turn", 0))      # turning
        raw[21] = float(active_actions.get("pitch", 0))     # pitching
        raw[22] = float(active_actions.get("jump", 0))      # jumping
        raw[23] = float(active_actions.get("crouch", 0))    # crouching
        raw[24] = float(active_actions.get("attack", 0))    # attacking

        return ids, raw

    def forward(self, ids: np.ndarray, raw: np.ndarray):
        """Forward pass with embedding lookup.

        1. Look up each ID in embedding table → 440 floats
        2. Apply ID mask (zero out disabled embeddings)
        3. Concatenate with masked raw floats → 465
        4. Hidden layer (32 ReLU)
        5. Output layer (23 softmax + action mask + min exploration)

        Returns: (probs, full_input, h1, ids_for_gradient)
        """
        # Embedding lookup: (110,) → (110, 4) → flatten → (440,)
        embedded = self.embeddings[ids]                    # (110, 4)
        # Apply ID mask: zero out disabled positions
        embedded = embedded * self.id_mask[:, None]        # broadcast (110,1) over (110,4)
        embedded_flat = embedded.flatten()                 # (440,)

        # Apply raw mask
        raw_masked = raw * self.raw_mask                   # (25,)

        # Concatenate: 440 + 25 = 465
        x = np.concatenate([embedded_flat, raw_masked])    # (465,)

        # Hidden layer
        h1 = x @ self.w1 + self.b1                        # (32,)
        h1 = np.maximum(0, h1)                             # ReLU

        # Output + action mask + softmax
        logits = h1 @ self.w2 + self.b2                    # (23,)
        logits = np.where(self.action_mask > 0, logits, -1e9)
        logits -= np.max(logits)
        exp_l = np.exp(logits)
        probs = exp_l / (exp_l.sum() + 1e-8)

        # Minimum exploration for enabled actions
        num_enabled = self.action_mask.sum()
        if num_enabled > 0 and self.min_exploration > 0:
            probs = np.where(self.action_mask > 0,
                             np.maximum(probs, self.min_exploration), 0.0)
            probs = probs / (probs.sum() + 1e-8)

        return probs, x, h1

    def choose_action(self, obs: dict, grid_blocks: list, active_actions: dict = None) -> int:
        """Full pipeline: observe → encode → embed → forward → sample.

        Returns: action index (0-22).
        """
        ids, raw = self._encode(obs, grid_blocks, active_actions)
        probs, x, h1 = self.forward(ids, raw)

        action = np.random.choice(NUM_ACTIONS, p=probs)

        # Save for REINFORCE
        self._ep_inputs.append(x)
        self._ep_id_indices.append(ids.copy())
        self._ep_h1s.append(h1)
        self._ep_probs.append(probs.copy())
        self._ep_actions.append(action)

        self.total_ticks += 1
        return action

    def record_reward(self, reward: float):
        """Record reward for the current timestep."""
        self._ep_rewards.append(reward)

    def update(self, gamma: float = 0.99) -> float:
        """REINFORCE update. Updates w1, b1, w2, b2, AND embeddings."""
        if not self._ep_rewards:
            return 0.0

        T = min(len(self._ep_rewards), len(self._ep_actions))

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
        d_embed = np.zeros_like(self.embeddings)

        for t in range(T):
            x = self._ep_inputs[t]
            h1 = self._ep_h1s[t]
            probs = self._ep_probs[t]
            action = self._ep_actions[t]
            ids = self._ep_id_indices[t]
            G = returns[t]

            # Output gradient
            d_logits = -probs.copy()
            d_logits[action] += 1.0
            d_logits *= G

            # Layer 2 (hidden → output)
            dw2 += np.outer(h1, d_logits)
            db2 += d_logits

            # Backprop to hidden
            d_h1 = d_logits @ self.w2.T
            d_h1 *= (h1 > 0).astype(np.float32)  # ReLU grad

            # Layer 1 (input → hidden)
            dw1 += np.outer(x, d_h1)
            db1 += d_h1

            # Backprop to embeddings
            # Gradient w.r.t. input x: d_x = d_h1 @ w1.T  (465,)
            d_x = d_h1 @ self.w1.T
            # First 440 floats are embedded IDs (110 × 4)
            d_embedded = d_x[:NUM_ID_INPUTS * EMBED_DIM].reshape(NUM_ID_INPUTS, EMBED_DIM)
            # Apply ID mask (only update enabled embeddings)
            d_embedded = d_embedded * self.id_mask[:, None]
            # Accumulate into embedding table rows
            for i in range(NUM_ID_INPUTS):
                if self.id_mask[i] > 0:
                    d_embed[ids[i]] += d_embedded[i]

        # Apply gradients
        scale = self.lr / max(T, 1)
        self.w2 += scale * dw2
        self.b2 += scale * db2
        self.w1 += scale * dw1
        self.b1 += scale * db1
        self.embeddings += scale * d_embed

        self.episodes_trained += 1

        # Clear episode buffer
        self._ep_inputs.clear()
        self._ep_id_indices.clear()
        self._ep_h1s.clear()
        self._ep_probs.clear()
        self._ep_actions.clear()
        self._ep_rewards.clear()

        return 0.0

    def save(self, directory: str):
        """Save weights, embeddings, vocab, history."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        np.savez(str(d / "weights.npz"),
                 w1=self.w1, b1=self.b1,
                 w2=self.w2, b2=self.b2,
                 embeddings=self.embeddings)

        with open(d / "vocab.json", "w") as f:
            json.dump({
                "block_vocab": self.block_vocab,
                "next_block_id": self.next_block_id,
                "input_level": self.input_level,
                "action_level": self.action_level,
            }, f, indent=2)

        with open(d / "history.json", "w") as f:
            json.dump({
                "name": self.name,
                "episodes_trained": self.episodes_trained,
                "total_ticks": self.total_ticks,
                "episode_history": self.episode_history,
                "vocab_size": self.next_block_id,
            }, f, indent=2)

    def load(self, directory: str):
        """Load saved agent state."""
        d = Path(directory)

        data = np.load(str(d / "weights.npz"))
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
        self.embeddings = data["embeddings"]

        with open(d / "vocab.json") as f:
            v = json.load(f)
            self.block_vocab = v["block_vocab"]
            self.next_block_id = v["next_block_id"]
            if "input_level" in v:
                self.input_level = v["input_level"]
                self.action_level = v["action_level"]

        if (d / "history.json").exists():
            with open(d / "history.json") as f:
                h = json.load(f)
                self.episodes_trained = h.get("episodes_trained", 0)
                self.total_ticks = h.get("total_ticks", 0)
                self.episode_history = h.get("episode_history", [])

    def set_levels(self, input_level: int, action_level: int):
        """Change mask levels. Network weights stay the same."""
        self.input_level = input_level
        self.action_level = action_level
        self.id_mask, self.raw_mask, self.action_mask = \
            _build_masks(input_level, action_level)

    def record_episode_stats(self, episode_data: dict):
        """Record full episode stats."""
        self.episode_history.append(episode_data)

    def __repr__(self):
        return (f"Brainstem('{self.name}', {NETWORK_INPUT_DIM}->{HIDDEN}->{NUM_ACTIONS}, "
                f"embed={EMBED_DIM}x{MAX_VOCAB}, "
                f"vocab={self.next_block_id}, ep={self.episodes_trained})")
