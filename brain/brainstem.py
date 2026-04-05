"""Brainstem — Stage 1 brain. 3-hidden-layer neural network.

Zero hand-crafted features. The agent receives:
    - Raw block IDs as integers (block type 0, 1, 2, ...) — no names, no categories
    - Raw internal numbers: health, food_level, x, y, z, yaw, pitch
    - The agent must learn what everything means by itself

Vision:
    7×7 grid, 2 layers (floor + eye level) = 98 blocks.
    Each block is an integer ID. The agent builds its own "names"
    (object1, object2, ...) through learned embeddings.

    Block IDs are assigned in order of first encounter. Each agent
    maintains its own block vocabulary — they independently learn
    what each block type means.

Internal state:
    7 raw numbers: health, food_level, x, y, z, yaw, pitch
    No normalization — the network learns the scales.

Actions (keyboard-like):
    0: W     — move forward
    1: S     — move backward
    2: A     — strafe left
    3: D     — strafe right
    4: ←     — turn left
    5: →     — turn right
    6: space — jump
    7: E     — use/interact/eat
    8: (nothing) — stand still

Architecture:
    Input (98 block IDs embedded + 7 internal = variable)
    → Hidden 1 (128 ReLU)
    → Hidden 2 (64 ReLU)
    → Output (9 softmax)

    Each block = 1 raw integer ID. 98 blocks + 7 internal = 105 inputs.

Learning:
    REINFORCE with per-timestep gradients.
    Each agent saves its own weights, vocabulary, and training history.
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

# Vision
GRID_SIZE = 7 * 7 * 2       # 98 blocks (floor + eye level)
HOTBAR_SLOTS = 9             # 9 hotbar item IDs
INTERNAL_DIM = 7 + HOTBAR_SLOTS + 1  # health,food,x,y,z,yaw,pitch + 9 hotbar + current slot = 17
INPUT_DIM = GRID_SIZE + INTERNAL_DIM  # 98 + 17 = 115

# Network
HIDDEN1 = 128
HIDDEN2 = 64


class Brainstem:
    """3-hidden-layer neural network brain with learned block embeddings.

    No hand-crafted features. Each agent has its own:
        - Block vocabulary (maps block names → integer IDs)
        - Embedding matrix (maps IDs → learned vectors)
        - Network weights
        - Training history
    """

    def __init__(self, name: str = "agent", learning_rate: float = 0.001):
        self.name = name
        self.lr = learning_rate

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

        # Training history
        self.episodes_trained = 0
        self.total_ticks = 0
        self.reward_history = []   # avg reward per episode

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

        Each block name → integer ID. That's it. No features, no embedding.
        The network learns what each ID means through training.

        Returns:
            np.ndarray shape (98,) of float block IDs.
        """
        block_ids = []
        for block in grid_blocks[:GRID_SIZE]:
            bid = self._get_block_id(str(block))
            block_ids.append(float(bid))

        # Pad if grid is smaller than expected
        while len(block_ids) < GRID_SIZE:
            block_ids.append(0.0)

        return np.array(block_ids[:GRID_SIZE], dtype=np.float32)

    def encode_internal(self, obs: dict) -> np.ndarray:
        """Extract raw internal state — no normalization, no feature engineering.

        Returns 17 floats:
            [health, food, x, y, z, yaw, pitch,
             hotbar_0_id, hotbar_1_id, ..., hotbar_8_id,
             current_slot]

        Hotbar items are raw integer IDs (same vocab as vision blocks).
        The agent learns what items are and which slot has what.
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

        # Hotbar: 9 slots, each item gets an ID from the same vocab
        # Empty slot = -1 (the network learns what -1 means)
        for slot in range(HOTBAR_SLOTS):
            item_key = f"Hotbar_{slot}_item"
            item_name = obs.get(item_key, "")
            if item_name:
                state.append(float(self._get_block_id(str(item_name))))
            else:
                state.append(-1.0)  # empty slot

        # Which slot is currently selected
        state.append(float(obs.get("currentItemIndex", 0)))

        return np.array(state, dtype=np.float32)

    def forward(self, vision: np.ndarray, internal: np.ndarray):
        """Forward pass: input → h1 (ReLU) → h2 (ReLU) → output (softmax).

        Returns: (probs, input_vec, h1, h2)
        """
        x = np.concatenate([vision, internal])

        # Layer 1
        h1 = x @ self.w1 + self.b1
        h1 = np.maximum(0, h1)  # ReLU

        # Layer 2
        h2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0, h2)  # ReLU

        # Output layer + softmax
        logits = h2 @ self.w3 + self.b3
        logits -= np.max(logits)
        exp_l = np.exp(logits)
        probs = exp_l / (exp_l.sum() + 1e-8)

        return probs, x, h1, h2

    def choose_action(self, obs: dict, grid_blocks: list) -> int:
        """Full pipeline: observe → encode → forward → sample action.

        Stores everything needed for learning.
        Returns: action index (0-8).
        """
        vision = self.encode_vision(grid_blocks)
        internal = self.encode_internal(obs)
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
        self.reward_history.append(avg_reward)

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

        # Vocabulary
        with open(d / "vocab.json", "w") as f:
            json.dump({
                "block_vocab": self.block_vocab,
                "next_block_id": self.next_block_id,
            }, f, indent=2)

        # Training history
        with open(d / "history.json", "w") as f:
            json.dump({
                "name": self.name,
                "episodes_trained": self.episodes_trained,
                "total_ticks": self.total_ticks,
                "reward_history": self.reward_history,
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

        # History
        if (d / "history.json").exists():
            with open(d / "history.json") as f:
                hist = json.load(f)
                self.episodes_trained = hist.get("episodes_trained", 0)
                self.total_ticks = hist.get("total_ticks", 0)
                self.reward_history = hist.get("reward_history", [])

    def __repr__(self):
        return (f"Brainstem('{self.name}', input={INPUT_DIM}, "
                f"h1={HIDDEN1}, h2={HIDDEN2}, actions={NUM_ACTIONS}, "
                f"vocab={self.next_block_id}, episodes={self.episodes_trained})")
