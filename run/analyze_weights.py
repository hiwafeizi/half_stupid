"""Analyze brainstem weights — what's connected to what.

Usage:
    python run/analyze_weights.py
    python run/analyze_weights.py --agent Adam
"""

import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.brainstem import (Brainstem, EMBED_DIM, NUM_ID_INPUTS, NUM_RAW_INPUTS,
                              NETWORK_INPUT_DIM, HIDDEN, NUM_ACTIONS, ACTION_NAMES,
                              CONTEXT_WINDOW, CONTEXT_HIDDEN, CONTEXT_PER_TICK,
                              CONTEXT_INPUT, CONTEXT_RAW_PER_TICK)

SAVE_DIR = "run/checkpoints/stage1"
AGENT_NAMES = ["Adam", "Eve", "Cain", "Abel"]

KNOWN_FOOD = {
    "cooked_beef", "cooked_chicken", "cooked_fish", "cooked_porkchop",
    "cooked_mutton", "cooked_rabbit", "melon", "carrot", "baked_potato",
    "apple", "bread", "cookie", "pumpkin_pie", "golden_apple",
    "mushroom_stew", "beetroot_soup",
}
KNOWN_UNEATABLE = {
    "stone", "stick", "cobblestone", "dirt", "iron_ingot", "gold_ingot",
    "diamond", "coal", "bone", "feather", "string", "arrow",
    "wooden_sword", "wooden_pickaxe", "leather", "paper", "bowl",
}

RAW_NAMES = [
    "health", "food", "x", "y", "z", "yaw", "pitch",
    "s1_ct", "s2_ct", "s3_ct", "s4_ct", "s5_ct",
    "s6_ct", "s7_ct", "s8_ct", "s9_ct",
    "held_ct", "eating", "moving", "strafing", "turning",
    "pitching", "jumping", "crouching", "attacking",
    "a1_here", "a1_rx", "a1_rz", "a1_hp",
    "a2_here", "a2_rx", "a2_rz", "a2_hp",
    "a3_here", "a3_rx", "a3_rz", "a3_hp",
    "i1_here", "i1_rx", "i1_rz", "i1_qty",
    "i2_here", "i2_rx", "i2_rz", "i2_qty",
    "i3_here", "i3_rx", "i3_rz", "i3_qty",
    "i4_here", "i4_rx", "i4_rz", "i4_qty",
    "i5_here", "i5_rx", "i5_rz", "i5_qty",
    # Spatial encoding (appended)
    "sin_yaw", "cos_yaw", "sin_pitch", "cos_pitch",
    "a1_dist", "a2_dist", "a3_dist",
    "a1_rsin", "a1_rcos", "a2_rsin", "a2_rcos", "a3_rsin", "a3_rcos",
    "i1_dist", "i2_dist", "i3_dist", "i4_dist", "i5_dist",
    "i1_rsin", "i1_rcos", "i2_rsin", "i2_rcos", "i3_rsin", "i3_rcos",
    "i4_rsin", "i4_rcos", "i5_rsin", "i5_rcos",
]


def analyze_agent(name):
    subdir = os.path.join(SAVE_DIR, name)
    if not os.path.exists(os.path.join(subdir, "weights.pt")):
        print(f"  {name}: no weights found")
        return

    brain = Brainstem(name=name)
    brain.load(subdir)

    vocab = brain.block_vocab
    embed = brain._np_embed    # (2048, 4)
    w1 = brain._np_w1          # (529, 32)
    b1 = brain._np_b1          # (32,)
    w2 = brain._np_w2          # (32, 23)
    b2 = brain._np_b2          # (23,)

    print(f"\n{'='*70}")
    print(f"  {name} — {brain.episodes_trained} ep, {brain.next_block_id} vocab")
    print(f"  Input: {brain.input_level}, Action: {brain.action_level}")
    print(f"{'='*70}")

    # ─── 1. EMBEDDINGS ───────────────────────────────────────
    food_embeds = {k: embed[v] for k, v in vocab.items() if k in KNOWN_FOOD}
    uneat_embeds = {k: embed[v] for k, v in vocab.items() if k in KNOWN_UNEATABLE}
    empty = embed[0]

    if food_embeds and uneat_embeds:
        food_avg = np.mean(list(food_embeds.values()), axis=0)
        uneat_avg = np.mean(list(uneat_embeds.values()), axis=0)
        dist = np.linalg.norm(food_avg - uneat_avg)
        print(f"\n  Embed dist food<->uneat: {dist:.2f} | EMPTY norm: {np.linalg.norm(empty):.2f}")
        print(f"  {'SEPARATED' if dist > 3 else 'OVERLAP'}")

    # ─── 2. W1: what inputs drive each hidden node ───────────
    # w1 is (465, 32). First 472 = embedded IDs (118 * 4), last 57 = raw.
    # For raw inputs: w1[472+j, node] = how much raw input j affects node
    # For embedded IDs: sum of |w1[id*4 : id*4+4, node]| = total impact of that ID slot

    print(f"\n  --- W1: INPUT -> HIDDEN (strongest connections) ---")

    # Compute total weight magnitude from each input GROUP to each node
    # Groups: vision(0-99), hotbar_ids(100-108), held_id(109), raw(25 floats)
    for node in range(HIDDEN):
        # Vision total impact: sum of absolute weights from 100*4=400 positions
        vis_impact = np.sum(np.abs(w1[0:400, node]))
        # Hotbar IDs impact: 9 slots * 4 dims = 36 positions
        hb_impact = np.sum(np.abs(w1[400:436, node]))
        # Held ID impact: 4 positions
        held_impact = np.sum(np.abs(w1[468:472, node]))
        # Each raw input impact
        raw_impacts = {RAW_NAMES[j]: abs(w1[472+j, node]) for j in range(NUM_RAW_INPUTS)}
        top_raw = sorted(raw_impacts.items(), key=lambda x: -x[1])[:3]

        top_raw_str = ", ".join(f"{k}={w1[472+list(raw_impacts.keys()).index(k), node]:+.2f}" for k, _ in top_raw)
        print(f"    n{node:2d} (b={b1[node]:+.2f}): vis={vis_impact:.1f} hb={hb_impact:.1f} held={held_impact:.1f} | {top_raw_str}")

    # ─── 3. W2: what hidden nodes drive each action ──────────
    print(f"\n  --- W2: HIDDEN -> ACTIONS (top 5 per action) ---")
    enabled = [j for j in range(NUM_ACTIONS) if brain.action_mask[j] > 0]
    for act_idx in enabled:
        weights = w2[:, act_idx]
        top5 = np.argsort(-np.abs(weights))[:5]
        parts = ", ".join(f"n{n}={weights[n]:+.2f}" for n in top5)
        print(f"    {ACTION_NAMES[act_idx]:15s} (b={b2[act_idx]:+.2f}): {parts}")

    # ─── 4. END-TO-END: which input groups most influence each action ───
    print(f"\n  --- END-TO-END: input group -> action influence ---")
    # For each action, compute: w2[:, action] @ w1.T -> (529,) influence vector
    # Then sum by group
    for act_idx in enabled:
        # Influence of each input position on this action (through hidden layer)
        # This is approximate — ignores ReLU nonlinearity
        influence = w2[:, act_idx] @ w1.T  # (465,)

        vis_inf = np.sum(np.abs(influence[0:400]))
        hb_inf = np.sum(np.abs(influence[400:436]))
        held_inf = np.sum(np.abs(influence[468:472]))
        raw_inf = {RAW_NAMES[j]: influence[472+j] for j in range(NUM_RAW_INPUTS)}
        top_raw = sorted(raw_inf.items(), key=lambda x: -abs(x[1]))[:4]
        top_str = ", ".join(f"{k}={v:+.1f}" for k, v in top_raw)

        print(f"    {ACTION_NAMES[act_idx]:15s}: vis={vis_inf:.0f} hb={hb_inf:.0f} held={held_inf:.0f} | {top_str}")

    # ─── 5. CONTEXT LAYER: W3 (160->64) + W4 (64->23) ────────
    w3 = brain._np_w3          # (205, 64)
    b3 = brain._np_b3          # (64,)
    w4 = brain._np_w4          # (64, 23)
    b4 = brain._np_b4          # (23,)

    ctx_total = np.abs(w3).sum() + np.abs(w4).sum()
    print(f"\n  --- CONTEXT LAYER (fires every {CONTEXT_WINDOW} ticks) ---")
    print(f"  Total weight magnitude: {ctx_total:.2f} {'(SILENT — not yet trained)' if ctx_total < 0.01 else ''}")

    if ctx_total > 0.01:
        # Context input layout:
        #   [0..204]  5 ticks × 41 per tick (32 hidden + 9 raw)
        #   [205..240] 9 hotbar item embeddings × 4 dims = 36
        CTX_RAW_NAMES = ["moving", "strafing", "turning", "pitching",
                         "jumping", "crouching", "attacking", "health", "food"]
        TICKS_END = CONTEXT_PER_TICK * CONTEXT_WINDOW  # 205
        from brain.brainstem import CONTEXT_HOTBAR_DIM

        # W3: which inputs drive each context hidden node
        print(f"\n  --- W3: CONTEXT INPUT -> CONTEXT HIDDEN ---")
        for cnode in range(CONTEXT_HIDDEN):
            # Per-tick breakdown
            tick_impacts = []
            for tick in range(CONTEXT_WINDOW):
                offset = tick * CONTEXT_PER_TICK
                tick_impact = np.sum(np.abs(w3[offset:offset+CONTEXT_PER_TICK, cnode]))
                tick_impacts.append(tick_impact)

            # Hidden vs raw vs hotbar breakdown
            hidden_total = 0
            raw_total = 0
            for tick in range(CONTEXT_WINDOW):
                offset = tick * CONTEXT_PER_TICK
                hidden_total += np.sum(np.abs(w3[offset:offset+HIDDEN, cnode]))
                raw_total += np.sum(np.abs(w3[offset+HIDDEN:offset+CONTEXT_PER_TICK, cnode]))
            hotbar_total = np.sum(np.abs(w3[TICKS_END:, cnode]))

            # Top raw signals across all ticks
            raw_scores = {}
            for ri, rname in enumerate(CTX_RAW_NAMES):
                total = sum(abs(w3[t * CONTEXT_PER_TICK + HIDDEN + ri, cnode])
                            for t in range(CONTEXT_WINDOW))
                raw_scores[rname] = total
            top_raw = sorted(raw_scores.items(), key=lambda x: -x[1])[:3]
            raw_str = ", ".join(f"{k}={v:.1f}" for k, v in top_raw)

            tick_str = " ".join(f"t{t}={v:.1f}" for t, v in enumerate(tick_impacts))
            print(f"    c{cnode:2d} (b={b3[cnode]:+.2f}): hidden={hidden_total:.1f} raw={raw_total:.1f} hbar={hotbar_total:.1f} | {tick_str} | {raw_str}")

        # W4: what context nodes drive each action
        print(f"\n  --- W4: CONTEXT HIDDEN -> ACTIONS ---")
        for act_idx in enabled:
            weights = w4[:, act_idx]
            top5 = np.argsort(-np.abs(weights))[:5]
            parts = ", ".join(f"c{n}={weights[n]:+.2f}" for n in top5)
            print(f"    {ACTION_NAMES[act_idx]:15s} (b={b4[act_idx]:+.2f}): {parts}")

        # Context end-to-end: which raw signals most influence each action
        print(f"\n  --- CONTEXT END-TO-END: raw signal -> action ---")
        for act_idx in enabled:
            ctx_e2e = w4[:, act_idx] @ w3.T  # (241,)
            # Sum raw signal influence across all ticks
            raw_inf = {}
            for ri, rname in enumerate(CTX_RAW_NAMES):
                total = sum(ctx_e2e[t * CONTEXT_PER_TICK + HIDDEN + ri]
                            for t in range(CONTEXT_WINDOW))
                raw_inf[rname] = total
            top = sorted(raw_inf.items(), key=lambda x: -abs(x[1]))[:4]
            top_str = ", ".join(f"{k}={v:+.1f}" for k, v in top)

            # Tick recency bias
            tick_inf = []
            for tick in range(CONTEXT_WINDOW):
                offset = tick * CONTEXT_PER_TICK
                tick_inf.append(np.sum(np.abs(ctx_e2e[offset:offset+CONTEXT_PER_TICK])))
            tick_str = " ".join(f"t{t}={v:.0f}" for t, v in enumerate(tick_inf))

            # Hotbar embedding influence
            hbar_inf = np.sum(np.abs(ctx_e2e[TICKS_END:]))

            print(f"    {ACTION_NAMES[act_idx]:15s}: {top_str} | {tick_str} | hbar={hbar_inf:.0f}")

        # Hotbar embedding influence summary
        print(f"\n  --- CONTEXT HOTBAR INFLUENCE (per slot) ---")
        for act_idx in enabled:
            ctx_e2e = w4[:, act_idx] @ w3.T
            slot_inf = []
            for slot in range(9):
                si = np.sum(np.abs(ctx_e2e[TICKS_END + slot * EMBED_DIM:TICKS_END + (slot+1) * EMBED_DIM]))
                slot_inf.append(si)
            slot_str = " ".join(f"s{s+1}={v:.0f}" for s, v in enumerate(slot_inf))
            print(f"    {ACTION_NAMES[act_idx]:15s}: {slot_str}")

    # ─── 6. EMBEDDING QUALITY: do food items cluster? ────────
    print(f"\n  --- EMBEDDING CLUSTERS ---")
    for item_name, item_embed in sorted(food_embeds.items()):
        e_str = ", ".join(f"{v:+.2f}" for v in item_embed)
        print(f"    FOOD  {item_name:20s}: [{e_str}]")
    for item_name, item_embed in sorted(uneat_embeds.items()):
        e_str = ", ".join(f"{v:+.2f}" for v in item_embed)
        print(f"    JUNK  {item_name:20s}: [{e_str}]")
    print(f"    EMPTY {'':20s}: [{', '.join(f'{v:+.2f}' for v in empty)}]")


def deep_analysis(name):
    """Which vision positions matter most, which nodes do what."""
    subdir = os.path.join(SAVE_DIR, name)
    if not os.path.exists(os.path.join(subdir, "weights.pt")):
        print(f"  {name}: no weights found")
        return

    brain = Brainstem(name=name)
    brain.load(subdir)
    vocab = brain.block_vocab
    reverse_vocab = {v: k for k, v in vocab.items()}
    embed = brain._np_embed
    w1 = brain._np_w1          # (529, 32)
    b1 = brain._np_b1
    w2 = brain._np_w2          # (32, 23)
    b2 = brain._np_b2

    print(f"\n{'='*70}")
    print(f"  {name} — DEEP ANALYSIS ({brain.episodes_trained} ep)")
    print(f"{'='*70}")

    # ─── 1. VISION: which of the 100 positions matter most for EAT ───
    # End-to-end influence of each vision position on use_eat (action 10)
    # influence = w2[:, 10] @ w1.T -> (529,)
    eat_influence = w2[:, 10] @ w1.T  # (465,)
    still_influence = w2[:, 22] @ w1.T

    # Each vision position is 4 floats (embedding dims). Sum absolute influence per position.
    print(f"\n  --- VISION: top 10 positions for EAT ---")
    print(f"  Layer guide: 0-24=below_feet, 25-49=floor, 50-74=eye, 75-99=above")
    vis_eat_impact = np.zeros(100)
    for pos in range(100):
        vis_eat_impact[pos] = np.sum(np.abs(eat_influence[pos*4 : pos*4+4]))
    top_vis = np.argsort(-vis_eat_impact)[:10]
    for pos in top_vis:
        layer = ["below_feet", "floor", "eye_level", "above_head"][pos // 25]
        local = pos % 25
        row, col = local // 5, local % 5
        sign = "+" if np.sum(eat_influence[pos*4:pos*4+4]) > 0 else "-"
        print(f"    pos {pos:3d} ({layer} r{row}c{col}): impact={vis_eat_impact[pos]:.1f} ({sign})")

    print(f"\n  --- VISION: top 10 positions for STAND STILL ---")
    vis_still_impact = np.zeros(100)
    for pos in range(100):
        vis_still_impact[pos] = np.sum(np.abs(still_influence[pos*4 : pos*4+4]))
    top_still = np.argsort(-vis_still_impact)[:10]
    for pos in top_still:
        layer = ["below_feet", "floor", "eye_level", "above_head"][pos // 25]
        local = pos % 25
        row, col = local // 5, local % 5
        sign = "+" if np.sum(still_influence[pos*4:pos*4+4]) > 0 else "-"
        print(f"    pos {pos:3d} ({layer} r{row}c{col}): impact={vis_still_impact[pos]:.1f} ({sign})")

    # ─── 2. VISION: heatmap by layer ────────────────────────
    print(f"\n  --- VISION EAT HEATMAP (5x5 per layer, higher=more eat drive) ---")
    for layer_idx, layer_name in enumerate(["below_feet", "floor", "eye_level", "above_head"]):
        print(f"    {layer_name}:")
        for row in range(5):
            vals = []
            for col in range(5):
                pos = layer_idx * 25 + row * 5 + col
                vals.append(f"{vis_eat_impact[pos]:5.0f}")
            print(f"      {' '.join(vals)}")

    # ─── 3. HIDDEN NODES: what does each node do? ───────────
    # For each node, compute: what action does it most strongly drive?
    print(f"\n  --- HIDDEN NODES: role of each node ---")
    enabled = [j for j in range(NUM_ACTIONS) if brain.action_mask[j] > 0]
    for node in range(HIDDEN):
        # What action does this node most influence?
        action_drives = {ACTION_NAMES[j]: w2[node, j] for j in enabled}
        top_action = max(action_drives, key=lambda k: abs(action_drives[k]))
        top_val = action_drives[top_action]

        # What inputs most activate this node?
        vis_total = np.sum(np.abs(w1[0:400, node]))
        raw_weights = w1[472:, node]
        top_raw_idx = np.argsort(-np.abs(raw_weights))[:2]
        raw_str = ", ".join(f"{RAW_NAMES[j]}={raw_weights[j]:+.1f}" for j in top_raw_idx)

        # Classify node role
        if abs(top_val) > 5:
            role = "STRONG"
        elif abs(top_val) > 2:
            role = "medium"
        else:
            role = "weak"

        print(f"    n{node:2d}: {role:6s} {top_action:>15s}={top_val:+.1f} | vis={vis_total:.0f} | {raw_str}")

    # ─── 4. EMBEDDING: which block types drive eating most? ──
    print(f"\n  --- WHICH BLOCKS DRIVE EATING (via vision)? ---")
    # For each known block, compute: if this block fills a high-impact vision position,
    # how much does it push toward eat vs still?
    # Use the top 5 vision positions for eat
    top5_vis = np.argsort(-vis_eat_impact)[:5]

    block_eat_scores = {}
    for block_name, block_id in vocab.items():
        if block_id == 0:
            continue
        block_embed = embed[block_id]  # (4,)
        # Score = dot product of embedding with eat influence at top positions
        score = 0
        for pos in top5_vis:
            inf = eat_influence[pos*4 : pos*4+4]
            score += np.dot(block_embed, inf)
        block_eat_scores[block_name] = score

    sorted_blocks = sorted(block_eat_scores.items(), key=lambda x: -x[1])
    print(f"    Most eat-promoting blocks (top 10):")
    for block_name, score in sorted_blocks[:10]:
        cat = "FOOD" if block_name in KNOWN_FOOD else "JUNK" if block_name in KNOWN_UNEATABLE else "BLOCK"
        print(f"      {cat:5s} {block_name:20s}: {score:+.1f}")
    print(f"    Most eat-suppressing blocks (bottom 5):")
    for block_name, score in sorted_blocks[-5:]:
        cat = "FOOD" if block_name in KNOWN_FOOD else "JUNK" if block_name in KNOWN_UNEATABLE else "BLOCK"
        print(f"      {cat:5s} {block_name:20s}: {score:+.1f}")

    # ─── 5. ALL ACTIONS: full breakdown ────────────────────────
    print(f"\n  --- ALL ACTIONS: what drives each one ---")
    enabled = [j for j in range(NUM_ACTIONS) if brain.action_mask[j] > 0]

    for act_idx in enabled:
        act_name = ACTION_NAMES[act_idx]
        inf = w2[:, act_idx] @ w1.T  # (529,)

        # Group influences
        vis_inf = np.sum(np.abs(inf[0:400]))
        hb_inf = np.sum(np.abs(inf[400:436]))
        held_inf = np.sum(np.abs(inf[436:440]))
        agent_inf = np.sum(np.abs(inf[440:452]))  # 3 agent IDs × 4 embed
        item_inf = np.sum(np.abs(inf[452:472]))    # 5 item IDs × 4 embed

        # Top raw inputs
        raw_inf = {RAW_NAMES[j]: inf[472+j] for j in range(NUM_RAW_INPUTS)}
        top_raw = sorted(raw_inf.items(), key=lambda x: -abs(x[1]))[:4]
        raw_str = ", ".join(f"{k}={v:+.1f}" for k, v in top_raw)

        # Which hidden nodes drive this action most
        act_weights = w2[:, act_idx]
        top_nodes = np.argsort(-np.abs(act_weights))[:3]
        node_str = ", ".join(f"n{n}={act_weights[n]:+.1f}" for n in top_nodes)

        # Top vision positions for this action
        vis_act = np.zeros(100)
        for pos in range(100):
            vis_act[pos] = np.sum(inf[pos*4:pos*4+4])  # signed, not abs
        top_vis = np.argsort(-np.abs(vis_act))[:3]
        layer_names = ["below", "floor", "eye", "above"]
        vis_str = ", ".join(
            f"{layer_names[p//25]}[{p%25//5},{p%5}]={'+'if vis_act[p]>0 else '-'}"
            for p in top_vis
        )

        # Top blocks that promote this action
        block_scores = {}
        for bname, bid in vocab.items():
            if bid == 0:
                continue
            be = embed[bid]
            score = sum(np.dot(be, inf[p*4:p*4+4]) for p in top_vis)
            block_scores[bname] = score
        top_blocks = sorted(block_scores.items(), key=lambda x: -x[1])[:3]
        block_str = ", ".join(f"{k}={v:+.0f}" for k, v in top_blocks)

        print(f"\n    {act_name}:")
        print(f"      Drive:  vis={vis_inf:.0f} hb={hb_inf:.0f} held={held_inf:.0f} agents={agent_inf:.0f} items={item_inf:.0f}")
        print(f"      Nodes:  {node_str}")
        print(f"      Raw:    {raw_str}")
        print(f"      Vision: {vis_str}")
        print(f"      Blocks: {block_str}")

    # ─── 6. CONTEXT NODES: what does each of the 64 context nodes do? ──
    w3 = brain._np_w3
    b3 = brain._np_b3
    w4 = brain._np_w4
    b4 = brain._np_b4
    ctx_total = np.abs(w3).sum() + np.abs(w4).sum()

    if ctx_total > 0.01:
        from brain.brainstem import (CONTEXT_WINDOW, CONTEXT_HIDDEN, CONTEXT_PER_TICK,
                                     CONTEXT_HOTBAR_DIM)
        TICKS_END = CONTEXT_PER_TICK * CONTEXT_WINDOW
        CTX_RAW_NAMES = ["moving", "strafing", "turning", "pitching",
                         "jumping", "crouching", "attacking", "health", "food"]

        print(f"\n  --- CONTEXT NODES: role of each context node ---")
        for cnode in range(CONTEXT_HIDDEN):
            # What action does this context node most influence?
            action_drives = {ACTION_NAMES[j]: w4[cnode, j] for j in enabled}
            top_action = max(action_drives, key=lambda k: abs(action_drives[k]))
            top_val = action_drives[top_action]

            # Input breakdown: hidden vs raw vs hotbar
            hidden_total = sum(np.sum(np.abs(w3[t*CONTEXT_PER_TICK:t*CONTEXT_PER_TICK+HIDDEN, cnode]))
                               for t in range(CONTEXT_WINDOW))
            raw_total = sum(np.sum(np.abs(w3[t*CONTEXT_PER_TICK+HIDDEN:t*CONTEXT_PER_TICK+CONTEXT_PER_TICK, cnode]))
                            for t in range(CONTEXT_WINDOW))
            hotbar_total = np.sum(np.abs(w3[TICKS_END:, cnode]))

            # Top raw signals
            raw_scores = {}
            for ri, rname in enumerate(CTX_RAW_NAMES):
                total = sum(abs(w3[t * CONTEXT_PER_TICK + HIDDEN + ri, cnode])
                            for t in range(CONTEXT_WINDOW))
                raw_scores[rname] = total
            top_raw = sorted(raw_scores.items(), key=lambda x: -x[1])[:3]
            raw_str = ", ".join(f"{k}={v:.1f}" for k, v in top_raw)

            # Tick recency: which tick matters most for this node?
            tick_weights = []
            for t in range(CONTEXT_WINDOW):
                tw = np.sum(np.abs(w3[t*CONTEXT_PER_TICK:(t+1)*CONTEXT_PER_TICK, cnode]))
                tick_weights.append(tw)
            heaviest_tick = np.argmax(tick_weights)
            tick_bias = "t%d" % heaviest_tick if max(tick_weights) > min(tick_weights) * 1.2 else "even"

            # Classify
            if abs(top_val) > 3:
                role = "STRONG"
            elif abs(top_val) > 1:
                role = "medium"
            elif abs(top_val) > 0.1:
                role = "weak"
            else:
                role = "dead"

            print(f"    c{cnode:2d}: {role:6s} {top_action:>15s}={top_val:+.2f} | "
                  f"hidden={hidden_total:.0f} raw={raw_total:.0f} hbar={hotbar_total:.0f} | "
                  f"tick={tick_bias} | {raw_str}")

        # ─── 7. CONTEXT: what drives each action (end-to-end) ──
        print(f"\n  --- CONTEXT ACTIONS: what drives each action through context ---")
        for act_idx in enabled:
            ctx_e2e = w4[:, act_idx] @ w3.T  # (241,)

            # Hidden vs raw vs hotbar total influence
            hidden_inf = sum(np.sum(np.abs(ctx_e2e[t*CONTEXT_PER_TICK:t*CONTEXT_PER_TICK+HIDDEN]))
                             for t in range(CONTEXT_WINDOW))
            raw_inf_total = sum(np.sum(np.abs(ctx_e2e[t*CONTEXT_PER_TICK+HIDDEN:t*CONTEXT_PER_TICK+CONTEXT_PER_TICK]))
                                for t in range(CONTEXT_WINDOW))
            hbar_inf = np.sum(np.abs(ctx_e2e[TICKS_END:]))

            # Top raw signals
            raw_inf = {}
            for ri, rname in enumerate(CTX_RAW_NAMES):
                total = sum(ctx_e2e[t * CONTEXT_PER_TICK + HIDDEN + ri]
                            for t in range(CONTEXT_WINDOW))
                raw_inf[rname] = total
            top = sorted(raw_inf.items(), key=lambda x: -abs(x[1]))[:4]
            raw_str = ", ".join(f"{k}={v:+.1f}" for k, v in top)

            # Top context nodes
            act_w = w4[:, act_idx]
            top_nodes = np.argsort(-np.abs(act_w))[:3]
            node_str = ", ".join(f"c{n}={act_w[n]:+.2f}" for n in top_nodes)

            print(f"\n    {ACTION_NAMES[act_idx]}:")
            print(f"      Drive:  hidden={hidden_inf:.0f} raw={raw_inf_total:.0f} hbar={hbar_inf:.0f}")
            print(f"      Nodes:  {node_str}")
            print(f"      Raw:    {raw_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default=None)
    parser.add_argument("--deep", action="store_true", help="Deep vision/node analysis")
    parser.add_argument("--save", action="store_true", help="Save output to txt file")
    args = parser.parse_args()

    agents = [args.agent] if args.agent else AGENT_NAMES

    if args.save:
        import io, sys
        output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = output

    for name in agents:
        if args.deep:
            deep_analysis(name)
        else:
            analyze_agent(name)

    if args.save:
        sys.stdout = old_stdout
        txt = output.getvalue()
        fname = f"run/checkpoints/stage1/analysis_{'deep_' if args.deep else ''}{agents[0] if len(agents)==1 else 'all'}.txt"
        with open(fname, "w") as f:
            f.write(txt)
        print(f"Saved to {fname}")
        print(txt[:200] + "...")


if __name__ == "__main__":
    main()
