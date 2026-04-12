"""Brainstem — Survival reflex module with learned embeddings + context layer.

NumPy forward pass (fast per-tick), PyTorch for gradient update (autograd).

Two-layer architecture:
    REFLEX (every tick):
        Embedding: 2048 vocab x 4 dims, shared across all block/item/agent IDs.
        118 ID inputs -> embedded to 472 floats + 57 raw floats = 529 total.
        Network: 529 -> 32 (ReLU) -> 23 logits
        Produces: reflex_logits (23) + hidden state (32)

    CONTEXT (every 5 ticks):
        Reads last 5 reflex hidden states as a sliding window.
        Network: 160 -> 64 (ReLU) -> 23 logits
        Output held constant between updates.
        Learns at half the reflex learning rate.

    ACTION SELECTION (every tick):
        final_logits = reflex_logits + context_logits
        softmax -> action probabilities -> sample

ID inputs (118):
    [0-99]     100 vision blocks (5x5x4)
    [100-108]  9 hotbar item IDs
    [109]      held item ID
    [110-112]  3 nearby agent name IDs (sorted by distance)
    [113-117]  5 nearby dropped item IDs (sorted by distance)

Raw inputs (85):
    [0-1]      health, food
    [2-6]      x, y, z, yaw, pitch
    [7-15]     9 hotbar counts
    [16]       held item count
    [17]       eating flag
    [18-24]    7 action flags (moving, strafing, turning, pitching, jumping, crouching, attacking)
    [25-36]    3 agents x (present, rel_x, rel_z, health)
    [37-56]    5 items x (present, rel_x, rel_z, quantity)
    [57-60]    sin_yaw, cos_yaw, sin_pitch, cos_pitch
    [61-63]    3 agent distances
    [64-69]    3 agents x (rel_angle_sin, rel_angle_cos)
    [70-74]    5 item distances
    [75-84]    5 items x (rel_angle_sin, rel_angle_cos)

Input masking levels (1-5), Action masking levels (1-6).
Per-agent saves: weights.pt, vocab.json, history.json
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

DEVICE = torch.device("cpu")

# ─── Actions ─────────────────────────────────────────────────
ACTIONS = [
    ("move", 1.0), ("move", -1.0), ("strafe", -1.0), ("strafe", 1.0),
    ("turn", -0.1), ("turn", 0.1), ("pitch", -0.1), ("pitch", 0.1),
    ("jump", 1), ("crouch", 1), ("use", 1), ("attack", 1),
    ("discardCurrentItem", 1),
    ("hotbar.1", 1), ("hotbar.2", 1), ("hotbar.3", 1), ("hotbar.4", 1),
    ("hotbar.5", 1), ("hotbar.6", 1), ("hotbar.7", 1), ("hotbar.8", 1),
    ("hotbar.9", 1),
    ("move", 0.0),
]
NUM_ACTIONS = len(ACTIONS)
ACTION_NAMES = [
    "W_forward", "S_backward", "A_strafe_left", "D_strafe_right",
    "turn_left", "turn_right", "look_up", "look_down",
    "jump", "crouch", "use_eat", "attack", "Q_throw",
    "hotbar_1", "hotbar_2", "hotbar_3", "hotbar_4",
    "hotbar_5", "hotbar_6", "hotbar_7", "hotbar_8", "hotbar_9",
    "stand_still",
]

# ─── Layout ──────────────────────────────────────────────────
GRID_W, GRID_H = 5, 4
GRID_SIZE = GRID_W * GRID_W * GRID_H  # 100
EMBED_DIM = 4
MAX_VOCAB = 2048
EMPTY_ID = 0
# ID inputs (each goes through embedding → 4 floats):
#   [0-99]     100 vision blocks (5×5×4)
#   [100-108]  9 hotbar item IDs
#   [109]      1 held item ID
#   [110-112]  3 nearby agent name IDs (sorted by distance)
#   [113-117]  5 nearby dropped item IDs (sorted by distance)
NUM_ID_INPUTS = 118

# Raw float inputs (pass through directly):
#   [0]        health
#   [1]        food
#   [2-6]      x, y, z, yaw, pitch
#   [7-15]     9 hotbar counts
#   [16]       held item count
#   [17]       eating flag
#   [18-24]    moving, strafing, turning, pitching, jumping, crouching, attacking
#   [25-28]    agent 1: present, rel_x, rel_z, health
#   [29-32]    agent 2: present, rel_x, rel_z, health
#   [33-36]    agent 3: present, rel_x, rel_z, health
#   [37-40]    item 1: present, rel_x, rel_z, quantity
#   [41-44]    item 2: present, rel_x, rel_z, quantity
#   [45-48]    item 3: present, rel_x, rel_z, quantity
#   [49-52]    item 4: present, rel_x, rel_z, quantity
#   [53-56]    item 5: present, rel_x, rel_z, quantity
# --- Spatial encoding (appended, old slots unchanged for backward compat) ---
#   [57]       sin(yaw)
#   [58]       cos(yaw)
#   [59]       sin(pitch)
#   [60]       cos(pitch)
#   [61-63]    3 agent distances
#   [64-69]    3 agents × (rel_angle_sin, rel_angle_cos)
#   [70-74]    5 item distances
#   [75-84]    5 items × (rel_angle_sin, rel_angle_cos)
NUM_RAW_INPUTS = 85

NETWORK_INPUT_DIM = NUM_ID_INPUTS * EMBED_DIM + NUM_RAW_INPUTS  # 118*4 + 85 = 557
HIDDEN = 32

# ─── Context layer ──────────────────────────────────────────
# Fires every CONTEXT_WINDOW ticks, reads recent reflex hidden states
# plus raw signals useful for temporal reasoning.
CONTEXT_WINDOW = 5                              # ticks between context updates
CONTEXT_HIDDEN = 64                             # context layer hidden nodes
# Per-tick context inputs: hidden state + action flags + health + food
#   32 hidden + 7 action flags + 2 (health, food) = 41 per tick
CONTEXT_RAW_PER_TICK = 9                        # 7 action flags + health + food
CONTEXT_PER_TICK = HIDDEN + CONTEXT_RAW_PER_TICK  # 32 + 9 = 41
# Appended once (last tick): 9 hotbar item embeddings × 4 dims = 36
CONTEXT_HOTBAR_DIM = 9 * EMBED_DIM              # 36
CONTEXT_INPUT = CONTEXT_PER_TICK * CONTEXT_WINDOW + CONTEXT_HOTBAR_DIM  # 41*5 + 36 = 241
CONTEXT_LR_SCALE = 0.5                          # learns at half the reflex rate

# ─── Masks ───────────────────────────────────────────────────
MASK_LEVELS = {
    # Level 1: health + food + eating flag (3 raw, 0 IDs = 3 active)
    1: {"vision": False, "hotbar_ids": False, "held_id": False,
        "health": True, "food": True, "position": False,
        "hotbar_counts": False, "held_count": False,
        "eating_flag": True, "other_flags": False, "entities": False},
    # Level 2: + held item ID + count (4 raw, 1 ID = 8 active)
    2: {"vision": False, "hotbar_ids": False, "held_id": True,
        "health": True, "food": True, "position": False,
        "hotbar_counts": False, "held_count": True,
        "eating_flag": True, "other_flags": False, "entities": False},
    # Level 3: + hotbar IDs + counts (13 raw, 10 IDs = 53 active)
    3: {"vision": False, "hotbar_ids": True, "held_id": True,
        "health": True, "food": True, "position": False,
        "hotbar_counts": True, "held_count": True,
        "eating_flag": True, "other_flags": False, "entities": False},
    # Level 4: + vision + action flags (20 raw, 110 IDs = 460 active)
    4: {"vision": True, "hotbar_ids": True, "held_id": True,
        "health": True, "food": True, "position": False,
        "hotbar_counts": True, "held_count": True,
        "eating_flag": True, "other_flags": True, "entities": False},
    # Level 5: + entities + position + spatial = everything (85 raw, 118 IDs = 557 active)
    5: {"vision": True, "hotbar_ids": True, "held_id": True,
        "health": True, "food": True, "position": True,
        "hotbar_counts": True, "held_count": True,
        "eating_flag": True, "other_flags": True, "entities": True},
}
ACTION_MASK_LEVELS = {
    1: [10, 22],
    2: [10, 13, 14, 15, 22],
    3: [10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    4: [0, 1, 4, 5, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    5: [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    6: list(range(NUM_ACTIONS)),
}


def _build_masks(input_level, action_level):
    il = max(l for l in MASK_LEVELS if l <= input_level)
    al = max(l for l in ACTION_MASK_LEVELS if l <= action_level)
    cfg = MASK_LEVELS[il]

    id_mask = np.zeros(NUM_ID_INPUTS, dtype=np.float32)
    if cfg["vision"]:       id_mask[0:100] = 1.0       # 100 vision blocks
    if cfg["hotbar_ids"]:   id_mask[100:109] = 1.0     # 9 hotbar IDs
    if cfg["held_id"]:      id_mask[109] = 1.0         # 1 held ID
    if cfg["entities"]:     id_mask[110:118] = 1.0     # 3 agent + 5 item IDs

    raw_mask = np.zeros(NUM_RAW_INPUTS, dtype=np.float32)
    if cfg["health"]:        raw_mask[0] = 1.0
    if cfg["food"]:          raw_mask[1] = 1.0
    if cfg["position"]:      raw_mask[2:7] = 1.0       # x,y,z,yaw,pitch
    if cfg["hotbar_counts"]: raw_mask[7:16] = 1.0      # 9 counts
    if cfg["held_count"]:    raw_mask[16] = 1.0
    if cfg["eating_flag"]:   raw_mask[17] = 1.0
    if cfg["other_flags"]:   raw_mask[18:25] = 1.0     # 7 action flags
    if cfg["entities"]:      raw_mask[25:57] = 1.0     # 3×4 agents + 5×4 items
    if cfg["position"]:      raw_mask[57:61] = 1.0     # sin/cos yaw + pitch
    if cfg["entities"]:      raw_mask[61:85] = 1.0     # distances + rel angles

    action_mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for idx in ACTION_MASK_LEVELS[al]:
        action_mask[idx] = 1.0

    return id_mask, raw_mask, action_mask


# ─── PyTorch net (only used in update()) ─────────────────────
class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Reflex path: fires every tick
        self.embedding = nn.Embedding(MAX_VOCAB, EMBED_DIM)
        self.fc1 = nn.Linear(NETWORK_INPUT_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, NUM_ACTIONS)
        # Context path: fires every CONTEXT_WINDOW ticks
        self.fc3 = nn.Linear(CONTEXT_INPUT, CONTEXT_HIDDEN)
        self.fc4 = nn.Linear(CONTEXT_HIDDEN, NUM_ACTIONS)
        with torch.no_grad():
            self.embedding.weight[EMPTY_ID].zero_()
            # Context: small random W3 so hidden activations are non-zero
            # (all-zero W3 would produce zero gradients through ReLU �� dead layer).
            # W4 starts at zero so context has no effect until W4 learns.
            nn.init.normal_(self.fc3.weight, std=0.01)
            self.fc3.bias.zero_()
            self.fc4.weight.zero_()
            self.fc4.bias.zero_()

    def forward(self, ids, raw, id_mask_t, raw_mask_t, action_mask_t,
                context_logits=None):
        """Reflex forward pass. Returns (log_probs, hidden_state).

        context_logits: (23,) tensor from context layer, added to reflex
                        logits before softmax. None = no context yet.
        """
        embedded = self.embedding(ids) * id_mask_t.unsqueeze(1)
        x = torch.cat([embedded.view(-1), raw * raw_mask_t])
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        if context_logits is not None:
            logits = logits + context_logits
        logits = logits.masked_fill(action_mask_t == 0, -1e9)
        return F.log_softmax(logits, dim=0), h

    def context_forward(self, hidden_buffer):
        """Context forward pass. Input: (160,) flattened hidden window."""
        ctx_h = F.relu(self.fc3(hidden_buffer))
        return self.fc4(ctx_h)


class Brainstem:
    def __init__(self, name="agent", learning_rate=0.001,
                 input_level=1, action_level=1, min_exploration=0.0002):
        self.name = name
        self.lr = learning_rate
        self.min_exploration = min_exploration
        self.input_level = input_level
        self.action_level = action_level

        self.id_mask, self.raw_mask, self.action_mask = _build_masks(input_level, action_level)

        self.block_vocab = {"EMPTY": EMPTY_ID}
        self.next_block_id = 1

        # PyTorch net (for update only)
        self.net = _Net().to(DEVICE)
        # Dual learning rates: context layer learns slower
        self.optimizer = torch.optim.Adam([
            {"params": [self.net.embedding.weight,
                        self.net.fc1.weight, self.net.fc1.bias,
                        self.net.fc2.weight, self.net.fc2.bias],
             "lr": learning_rate},
            {"params": [self.net.fc3.weight, self.net.fc3.bias,
                        self.net.fc4.weight, self.net.fc4.bias],
             "lr": learning_rate * CONTEXT_LR_SCALE},
        ])

        # NumPy copies of weights (for fast forward pass)
        self._sync_from_torch()

        # Context layer state: each tick stores hidden (32) + action flags (7) + health + food = 41
        self._context_buffer = np.zeros((CONTEXT_WINDOW, CONTEXT_PER_TICK), dtype=np.float32)
        self._tick_in_window = 0                 # 0..4, fires at CONTEXT_WINDOW
        self._context_logits = np.zeros(NUM_ACTIONS, dtype=np.float32)

        # Episode buffer
        self._ep_ids = []       # (118,) int arrays
        self._ep_raw = []       # (57,) float arrays
        self._ep_actions = []
        self._ep_rewards = []

        # Last tick (for debug JSON)
        self._last_ids = np.zeros(NUM_ID_INPUTS, dtype=np.int64)
        self._last_raw = np.zeros(NUM_RAW_INPUTS, dtype=np.float32)
        self._last_probs = np.zeros(NUM_ACTIONS, dtype=np.float32)

        self.episodes_trained = 0
        self.total_ticks = 0
        self.episode_history = []

    def _sync_from_torch(self):
        """Copy PyTorch weights to NumPy arrays for fast forward pass."""
        with torch.no_grad():
            # Reflex weights
            self._np_embed = self.net.embedding.weight.numpy().copy()
            self._np_w1 = self.net.fc1.weight.numpy().T.copy()   # (529, 32)
            self._np_b1 = self.net.fc1.bias.numpy().copy()       # (32,)
            self._np_w2 = self.net.fc2.weight.numpy().T.copy()   # (32, 23)
            self._np_b2 = self.net.fc2.bias.numpy().copy()       # (23,)
            # Context weights
            self._np_w3 = self.net.fc3.weight.numpy().T.copy()   # (160, 64)
            self._np_b3 = self.net.fc3.bias.numpy().copy()       # (64,)
            self._np_w4 = self.net.fc4.weight.numpy().T.copy()   # (64, 23)
            self._np_b4 = self.net.fc4.bias.numpy().copy()       # (23,)

    def _id(self, name):
        bid = self.block_vocab.get(name)
        if bid is not None:
            return bid
        if not name or name == "air":
            return EMPTY_ID
        if self.next_block_id < MAX_VOCAB:
            bid = self.next_block_id
            self.block_vocab[name] = bid
            self.next_block_id += 1
            return bid
        return MAX_VOCAB - 1

    def _encode(self, obs, grid_blocks, active_actions=None):
        if active_actions is None:
            active_actions = {}

        _id = self._id
        ids = np.zeros(NUM_ID_INPUTS, dtype=np.int64)
        raw = np.zeros(NUM_RAW_INPUTS, dtype=np.float32)

        # Vision: 100 blocks
        for i in range(min(len(grid_blocks), GRID_SIZE)):
            ids[i] = _id(grid_blocks[i])

        # Hotbar + held
        current_idx = int(obs.get("currentItemIndex", 0))
        held_ct = 0
        for slot in range(9):
            ct = obs.get(f"Hotbar_{slot}_size", 0)
            if ct:
                item = obs.get(f"Hotbar_{slot}_item", "")
                if item:
                    bid = _id(item)
                    ids[100 + slot] = bid
                    if slot == current_idx:
                        ids[109] = bid
                        held_ct = int(ct)

        # Nearby entities: 3 agent slots + 5 item slots
        nearby = obs.get("nearby", [])
        my_name = obs.get("Name", "")
        my_x = obs.get("XPos", 0.0)
        my_z = obs.get("ZPos", 0.0)

        agents = []
        items = []
        for entity in nearby:
            ename = entity.get("name", "")
            if ename == my_name:
                continue  # skip self
            if "life" in entity:
                # It's a player/mob
                agents.append(entity)
            elif "quantity" in entity:
                # It's a dropped item
                items.append(entity)

        # Sort by distance
        def dist(e):
            dx = e.get("x", 0) - my_x
            dz = e.get("z", 0) - my_z
            return dx * dx + dz * dz
        agents.sort(key=dist)
        items.sort(key=dist)

        # Fill 3 agent slots
        for slot in range(3):
            if slot < len(agents):
                a = agents[slot]
                ids[110 + slot] = _id(a.get("name", ""))
                raw[25 + slot * 4] = 1.0                      # present
                raw[26 + slot * 4] = a.get("x", 0) - my_x    # rel_x
                raw[27 + slot * 4] = a.get("z", 0) - my_z    # rel_z
                raw[28 + slot * 4] = a.get("life", 0)         # health

        # Fill 5 item slots
        for slot in range(5):
            if slot < len(items):
                it = items[slot]
                ids[113 + slot] = _id(it.get("name", ""))
                raw[37 + slot * 4] = 1.0                      # present
                raw[38 + slot * 4] = it.get("x", 0) - my_x   # rel_x
                raw[39 + slot * 4] = it.get("z", 0) - my_z   # rel_z
                raw[40 + slot * 4] = it.get("quantity", 0)    # quantity

        # Body + hotbar raw floats
        raw[0] = obs.get("Life", 0.0)
        raw[1] = obs.get("Food", 0.0)
        raw[2] = obs.get("XPos", 0.0)
        raw[3] = obs.get("YPos", 0.0)
        raw[4] = obs.get("ZPos", 0.0)
        raw[5] = obs.get("Yaw", 0.0)
        raw[6] = obs.get("Pitch", 0.0)
        for slot in range(9):
            raw[7 + slot] = float(obs.get(f"Hotbar_{slot}_size", 0))
        raw[16] = float(held_ct)
        raw[17] = float(active_actions.get("use", 0))
        raw[18] = float(active_actions.get("move", 0))
        raw[19] = float(active_actions.get("strafe", 0))
        raw[20] = float(active_actions.get("turn", 0))
        raw[21] = float(active_actions.get("pitch", 0))
        raw[22] = float(active_actions.get("jump", 0))
        raw[23] = float(active_actions.get("crouch", 0))
        raw[24] = float(active_actions.get("attack", 0))

        # ── Spatial encoding (appended at end, old slots untouched) ──
        yaw_rad = math.radians(obs.get("Yaw", 0.0))
        pitch_rad = math.radians(obs.get("Pitch", 0.0))
        raw[57] = math.sin(yaw_rad)
        raw[58] = math.cos(yaw_rad)
        raw[59] = math.sin(pitch_rad)
        raw[60] = math.cos(pitch_rad)

        # Agent distances + relative angles (where are they relative to my facing?)
        for slot in range(3):
            if slot < len(agents):
                a = agents[slot]
                dx = a.get("x", 0) - my_x
                dz = a.get("z", 0) - my_z
                distance = math.sqrt(dx * dx + dz * dz)
                # Angle from my facing direction to entity
                entity_angle = math.atan2(-dx, dz)  # Minecraft: +z=south, -x=east
                rel_angle = entity_angle - yaw_rad
                raw[61 + slot] = distance
                raw[64 + slot * 2] = math.sin(rel_angle)
                raw[65 + slot * 2] = math.cos(rel_angle)

        # Item distances + relative angles
        for slot in range(5):
            if slot < len(items):
                it = items[slot]
                dx = it.get("x", 0) - my_x
                dz = it.get("z", 0) - my_z
                distance = math.sqrt(dx * dx + dz * dz)
                entity_angle = math.atan2(-dx, dz)
                rel_angle = entity_angle - yaw_rad
                raw[70 + slot] = distance
                raw[75 + slot * 2] = math.sin(rel_angle)
                raw[76 + slot * 2] = math.cos(rel_angle)

        return ids, raw

    def _forward_np(self, ids, raw):
        """NumPy forward pass — fast, no autograd overhead.

        Reflex path runs every tick. Context path runs every
        CONTEXT_WINDOW ticks and holds its output in between.
        """
        # ── Reflex path (every tick) ──
        embedded = self._np_embed[ids]                       # (118, 4)
        embedded *= self.id_mask[:, None]                    # mask
        x = np.concatenate([embedded.ravel(), raw * self.raw_mask])  # (529,)
        self._last_x = x                                    # cache for debug

        h = x @ self._np_w1 + self._np_b1                   # (32,)
        h = np.maximum(0, h)                                 # ReLU
        self._last_hidden = h                                # cache for buffer

        reflex_logits = h @ self._np_w2 + self._np_b2       # (23,)

        # ── Context path (every CONTEXT_WINDOW ticks) ──
        # Pack: hidden (32) + action flags (7) + health + food = 41 per tick
        ctx_raw = np.concatenate([h, raw[18:25], raw[0:2]])  # (41,)
        self._context_buffer[self._tick_in_window] = ctx_raw
        self._tick_in_window += 1
        if self._tick_in_window >= CONTEXT_WINDOW:
            # Append hotbar embeddings from current tick (9 items × 4 dims = 36)
            hotbar_embeds = self._np_embed[ids[100:109]].ravel()  # (36,)
            ctx_input = np.concatenate([
                self._context_buffer.ravel(),                # (205,)
                hotbar_embeds,                               # (36,)
            ])                                               # (241,)
            ctx_h = ctx_input @ self._np_w3 + self._np_b3   # (64,)
            ctx_h = np.maximum(0, ctx_h)                     # ReLU
            self._context_logits = ctx_h @ self._np_w4 + self._np_b4  # (23,)
            self._tick_in_window = 0

        # ── Combine reflex + context ──
        logits = reflex_logits + self._context_logits
        logits = np.where(self.action_mask > 0, logits, -1e9)
        logits -= logits.max()
        exp_l = np.exp(logits)
        probs = exp_l / (exp_l.sum() + 1e-8)

        # Min exploration
        if self.min_exploration > 0:
            probs = np.where(self.action_mask > 0,
                             np.maximum(probs, self.min_exploration), 0.0)
            probs /= probs.sum() + 1e-8

        return probs

    def choose_action(self, obs, grid_blocks, active_actions=None):
        ids, raw = self._encode(obs, grid_blocks, active_actions)
        probs = self._forward_np(ids, raw)

        action = np.random.choice(NUM_ACTIONS, p=probs)

        self._ep_ids.append(ids)
        self._ep_raw.append(raw)
        self._ep_actions.append(action)

        self._last_ids = ids
        self._last_raw = raw
        self._last_probs = probs

        self.total_ticks += 1
        return action

    def record_reward(self, reward):
        self._ep_rewards.append(reward)

    def update(self, gamma=0.99):
        """PyTorch REINFORCE update. Replays stored obs through torch net.

        Replays both reflex and context paths with the same 5-tick timing
        used during the actual episode.
        """
        T = min(len(self._ep_rewards), len(self._ep_actions))
        if T == 0:
            self._ep_ids.clear()
            self._ep_raw.clear()
            self._ep_actions.clear()
            self._ep_rewards.clear()
            return 0.0

        # Discounted returns (numpy)
        returns_np = np.zeros(T, dtype=np.float32)
        R = 0.0
        for t in range(T - 1, -1, -1):
            R = self._ep_rewards[t] + gamma * R
            returns_np[t] = R
        if T > 1:
            returns_np = (returns_np - returns_np.mean()) / (returns_np.std() + 1e-8)

        # Replay through PyTorch net for gradients
        id_mask_t = torch.from_numpy(self.id_mask)
        raw_mask_t = torch.from_numpy(self.raw_mask)
        action_mask_t = torch.from_numpy(self.action_mask)
        returns_t = torch.from_numpy(returns_np)

        # Context replay state (mirrors the NumPy forward pass timing)
        # Use a list to avoid in-place tensor modifications that break autograd.
        ctx_frames = []
        context_logits = None
        tick_in_window = 0

        loss = torch.tensor(0.0)
        for t in range(T):
            ids_t = torch.from_numpy(self._ep_ids[t])
            raw_t = torch.from_numpy(self._ep_raw[t])
            log_probs, h = self.net(ids_t, raw_t, id_mask_t, raw_mask_t,
                                    action_mask_t, context_logits)

            # Pack same signals as _forward_np: hidden + action flags + health/food
            # No detach — context gradient flows back through fc1 and embeddings
            ctx_raw = torch.cat([h, raw_t[18:25], raw_t[0:2]])
            ctx_frames.append(ctx_raw)
            tick_in_window += 1
            if tick_in_window >= CONTEXT_WINDOW:
                window = torch.stack(ctx_frames[-CONTEXT_WINDOW:])
                # Append hotbar embeddings from current tick (gradient flows to embeddings)
                hotbar_ids_t = ids_t[100:109]
                hotbar_embeds = self.net.embedding(hotbar_ids_t).view(-1)  # (36,)
                ctx_input = torch.cat([window.view(-1), hotbar_embeds])    # (241,)
                context_logits = self.net.context_forward(ctx_input)
                tick_in_window = 0

            loss -= log_probs[self._ep_actions[t]] * returns_t[t]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Sync numpy weights from updated torch
        self._sync_from_torch()

        # Reset context state for next episode
        self._context_buffer.fill(0)
        self._tick_in_window = 0
        self._context_logits.fill(0)

        self.episodes_trained += 1
        self._ep_ids.clear()
        self._ep_raw.clear()
        self._ep_actions.clear()
        self._ep_rewards.clear()
        return 0.0

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"net": self.net.state_dict(),
                     "optimizer": self.optimizer.state_dict()},
                    str(d / "weights.pt"))
        with open(d / "vocab.json", "w") as f:
            json.dump({"block_vocab": self.block_vocab,
                        "next_block_id": self.next_block_id,
                        "input_level": self.input_level,
                        "action_level": self.action_level}, f, indent=2)
        with open(d / "history.json", "w") as f:
            json.dump({"name": self.name,
                        "episodes_trained": self.episodes_trained,
                        "total_ticks": self.total_ticks,
                        "episode_history": self.episode_history,
                        "vocab_size": self.next_block_id}, f, indent=2)

    def load(self, directory):
        d = Path(directory)
        ckpt = torch.load(str(d / "weights.pt"), map_location=DEVICE)
        saved_state = ckpt["net"]

        # Handle loading old checkpoints:
        # - Missing keys (fc3, fc4 from old saves) → keep current init (zeros)
        # - Shape mismatch (e.g. 465→32 to 529→32) → pad with zeros
        current_state = self.net.state_dict()
        for key in current_state:
            if key not in saved_state:
                # New layer not in old checkpoint — keep current init
                # (fc3.weight has small random init, others are zero)
                saved_state[key] = current_state[key]
                continue
            saved_shape = saved_state[key].shape
            current_shape = current_state[key].shape
            if saved_shape != current_shape:
                padded = current_state[key].clone()
                padded.zero_()
                slices = tuple(slice(0, min(s, c)) for s, c in zip(saved_shape, current_shape))
                padded[slices] = saved_state[key][slices]
                saved_state[key] = padded

        self.net.load_state_dict(saved_state)
        # If fc3 weights are all zero (dead layer), re-init with small random
        with torch.no_grad():
            if self.net.fc3.weight.abs().sum() == 0:
                nn.init.normal_(self.net.fc3.weight, std=0.01)
        # Reset optimizer with dual learning rates
        self.optimizer = torch.optim.Adam([
            {"params": [self.net.embedding.weight,
                        self.net.fc1.weight, self.net.fc1.bias,
                        self.net.fc2.weight, self.net.fc2.bias],
             "lr": self.lr},
            {"params": [self.net.fc3.weight, self.net.fc3.bias,
                        self.net.fc4.weight, self.net.fc4.bias],
             "lr": self.lr * CONTEXT_LR_SCALE},
        ])
        self._sync_from_torch()
        with open(d / "vocab.json") as f:
            v = json.load(f)
            self.block_vocab = v["block_vocab"]
            self.next_block_id = v["next_block_id"]
            if "input_level" in v:
                self.set_levels(v["input_level"], v["action_level"])
        if (d / "history.json").exists():
            with open(d / "history.json") as f:
                h = json.load(f)
                self.episodes_trained = h.get("episodes_trained", 0)
                self.total_ticks = h.get("total_ticks", 0)
                self.episode_history = h.get("episode_history", [])

    def set_levels(self, input_level, action_level):
        self.input_level = input_level
        self.action_level = action_level
        self.id_mask, self.raw_mask, self.action_mask = _build_masks(input_level, action_level)

    def record_episode_stats(self, data):
        self.episode_history.append(data)

    def __repr__(self):
        return (f"Brainstem('{self.name}', "
                f"reflex={NETWORK_INPUT_DIM}->{HIDDEN}->{NUM_ACTIONS}, "
                f"context={CONTEXT_INPUT}->{CONTEXT_HIDDEN}->{NUM_ACTIONS} "
                f"every {CONTEXT_WINDOW}t, "
                f"embed={EMBED_DIM}x{MAX_VOCAB}, "
                f"vocab={self.next_block_id}, ep={self.episodes_trained})")
