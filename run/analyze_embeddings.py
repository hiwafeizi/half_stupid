"""Analyze embedding clusters — distances between all item categories.

Usage:
    python run/analyze_embeddings.py
    python run/analyze_embeddings.py --agent Eve
    python run/analyze_embeddings.py --save
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.brainstem import Brainstem, EMBED_DIM

SAVE_DIR = "run/checkpoints/stage1"
AGENT_NAMES = ["Adam", "Eve", "Cain", "Abel"]

# Categories
CATEGORIES = {
    "FOOD": {
        "cooked_beef", "cooked_chicken", "cooked_fish", "cooked_porkchop",
        "cooked_mutton", "cooked_rabbit", "melon", "carrot", "baked_potato",
        "apple", "bread", "cookie", "pumpkin_pie", "golden_apple",
        "mushroom_stew", "beetroot_soup",
    },
    "POISON": {
        "spider_eye", "rotten_flesh", "poisonous_potato", "raw_chicken",
    },
    "JUNK": {
        "stone", "stick", "cobblestone", "dirt", "iron_ingot", "gold_ingot",
        "diamond", "coal", "bone", "feather", "string", "arrow",
        "wooden_sword", "wooden_pickaxe", "leather", "paper",
    },
    "TERRAIN": {
        "grass", "dirt", "air", "log", "leaves", "fence", "gravel",
        "bedrock", "quartz_block", "tripwire",
    },
    "DANGER": {
        "lava", "flowing_lava", "fire",
    },
    "DECORATION": {
        "red_flower", "yellow_flower", "double_plant", "tallgrass",
    },
    "CONTAINER": {
        "bowl",
    },
    "AGENT": set(),  # filled dynamically per agent
}


def analyze_agent(name):
    subdir = os.path.join(SAVE_DIR, name)
    if not os.path.exists(os.path.join(subdir, "weights.pt")):
        print("  %s: no weights found" % name)
        return

    b = Brainstem(name=name)
    b.load(subdir)
    embed = b._np_embed
    vocab = b.block_vocab

    # Find agent names in vocab (other players) — each gets own category
    cats = dict(CATEGORIES)
    del cats["AGENT"]
    for k in vocab:
        if k in AGENT_NAMES and k != name:
            cats[k] = {k}

    print("")
    print("=" * 70)
    print("  %s — %d ep, %d vocab" % (name, b.episodes_trained, b.next_block_id))
    print("=" * 70)

    # Collect embeddings per category
    cat_embeds = {}
    cat_items = {}
    for cat_name, cat_set in cats.items():
        items = {}
        for item_name, item_id in vocab.items():
            if item_name in cat_set:
                items[item_name] = embed[item_id]
        if items:
            cat_embeds[cat_name] = np.mean(list(items.values()), axis=0)
            cat_items[cat_name] = items

    # Add EMPTY
    cat_embeds["EMPTY"] = embed[0]
    cat_items["EMPTY"] = {"EMPTY": embed[0]}

    # Print category averages
    fmt = lambda v: ", ".join("%+.2f" % x for x in v)
    print("\n  --- CATEGORY AVERAGES ---")
    for cat_name in sorted(cat_embeds):
        n = len(cat_items.get(cat_name, {}))
        print("    %-12s (%2d): [%s]" % (cat_name, n, fmt(cat_embeds[cat_name])))

    # Distance matrix between all categories
    cat_names = sorted(cat_embeds.keys())
    print("\n  --- DISTANCE MATRIX ---")
    header = "%-12s" % "" + "".join("  %-8s" % c[:8] for c in cat_names)
    print("    " + header)
    for i, c1 in enumerate(cat_names):
        row = "%-12s" % c1
        for j, c2 in enumerate(cat_names):
            d = np.linalg.norm(cat_embeds[c1] - cat_embeds[c2])
            row += "  %8.2f" % d
        print("    " + row)

    # Where does each poison item sit relative to categories?
    print("\n  --- POISON ITEMS: closest category ---")
    for pname, pe in sorted(cat_items.get("POISON", {}).items()):
        dists = {}
        for cat_name, cat_avg in cat_embeds.items():
            if cat_name != "POISON":
                dists[cat_name] = np.linalg.norm(pe - cat_avg)
        closest = min(dists, key=dists.get)
        dist_str = ", ".join("%s=%.1f" % (k, v) for k, v in sorted(dists.items(), key=lambda x: x[1])[:4])
        print("    %-20s -> %s (%s)" % (pname, closest, dist_str))

    # Per-item embeddings within each category + spread
    print("\n  --- PER-CATEGORY ITEMS + SPREAD ---")
    agent_cats = [k for k in cat_embeds if k in AGENT_NAMES]
    for cat_name in ["FOOD", "POISON", "JUNK", "DANGER"] + sorted(agent_cats):
        items = cat_items.get(cat_name, {})
        if not items:
            continue
        avg = cat_embeds[cat_name]
        spread = np.mean([np.linalg.norm(e - avg) for e in items.values()])
        print("    %s (spread=%.2f):" % (cat_name, spread))
        for iname, ie in sorted(items.items()):
            d_own = np.linalg.norm(ie - avg)
            # Find closest OTHER category
            other_dists = {}
            for cn, ca in cat_embeds.items():
                if cn != cat_name:
                    other_dists[cn] = np.linalg.norm(ie - ca)
            closest_other = min(other_dists, key=other_dists.get)
            print("      %-20s [%s]  own=%.1f  closest=%s(%.1f)" %
                  (iname, fmt(ie), d_own, closest_other, other_dists[closest_other]))


def main():
    parser = argparse.ArgumentParser(description="Analyze embedding clusters")
    parser.add_argument("--agent", type=str, default=None)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    agents = [args.agent] if args.agent else AGENT_NAMES

    if args.save:
        import io
        output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = output

    for name in agents:
        analyze_agent(name)

    if args.save:
        sys.stdout = old_stdout
        txt = output.getvalue()
        fname = "run/checkpoints/stage1/embeddings_%s.txt" % (agents[0] if len(agents) == 1 else "all")
        with open(fname, "w") as f:
            f.write(txt)
        print("Saved to %s" % fname)
        print(txt[:300] + "...")


if __name__ == "__main__":
    main()
