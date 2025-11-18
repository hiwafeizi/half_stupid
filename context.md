
# **Project: Half_Stupid**

An experimental long-form simulation where artificial agents evolve cognition, memory, behavior, and internal representations inside a Minecraft-based world.

The project is inspired by ideas from cognitive science, neuroscience, evolutionary psychology, emergent communication, and artificial life.

The long-term goal is to create agents that **develop their own behaviors, strategies, concepts, and social structures** with minimal hand-crafted rules.

---

## **High-Level Vision**

We want to create a *bottom-up* artificial cognition system:

* Agents start with **no knowledge of the world**
* Only a few **basic drives** (hunger, safety, curiosity, exploration, simple bonding)
* A minimal set of **cognitive systems** such as:
  * Sensory perception
  * Short-term memory
  * Long-term memory
  * Spatial memory
  * Fast reactive system (low computation)
  * Slow deliberative system (higher computation)
  * Simple internal state (emotions/valence)
* Agents discover, learn, and build their own understanding
* Over generations, behavior evolves into something more complex

The goal is not realism but  **emergent complexity** .

---

## **Core Ideas**

### **1. Agents do not start intelligent**

They begin closer to animal intelligence.

Language, social behavior, and planning should emerge slowly through:

* learning
* trial and error
* evolution
* generational memory
* social imitation (eventually)

### **2. No hand-crafted human concepts**

We avoid injecting human labels like “tree”, “food”, “enemy”, etc.

Agents must learn:

* what things are
* how to use them
* how to name them
* which objects matter
* how to behave
* how to interact
* what to value

This is key to creating  **zero-to-something intelligence** , not “fine-tuned humans”.

### **3. The Paradise/Origin Scenario**

Our initial world concept:

* Two agents (“Adam & Eve”) in a protected paradise
* Simple survival conditions, stable resources
* A central **temple** with symbolic meaning (emergent, not given)
* A hidden **maze** with an intentional final choice:
  * Destroy the temple
  * Leave peacefully
* A hidden **exit door** to the outer world
* Door stays open briefly—others follow only by luck
* Generations are possible (offspring inherit no knowledge)

Interpretation and mythology must  **emerge** , not be pre-scripted.

### **4. Evolution as acceleration**

To reach interesting behaviors:

* Agents compete for survival
* Those who adapt persist
* Weak strategies die
* Cognitive systems may be improved generation by generation
* Not by training huge LLMs, but by raw behavior loops and simple model updates

### **5. The world is the teacher**

Learning emerges through:

* curiosity
* reward
* reinforcement
* surprise
* prediction errors
* social interactions
* exploration
* environmental pressure

---

## **Technical Foundations**

### **Environment**

* **Minecraft Java Edition**
* **Malmo 0.37.0** (Forge 1.11.2 mod)
* Highly programmable missions, blocks, structures, and agent interfaces

### **Agent Code**

* Python (via `MalmoPython.pyd`)
* No dependency on large LLMs

  (maybe small networks for perception or planning, but human-like language is not injected)

### **Cognitive Modules (current plan)**

* Vision → grid or ray observations
* Short-term memory buffer
* Long-term associative memory
* Spatial map memory
* Fast reflexive system (simple logic)
* Slow deliberate system (small NN or rule engine)
* Emotion-like scalar values (valence system)
* Curiosity system (intrinsic motivation)

### **Current Scope**

We are NOT building a human brain.

We want a **loosely inspired cognitive architecture** that produces “alive-feeling” behavior.

---

## **Project Philosophy**

* Start simple
* Let complexity emerge, don’t handcraft it
* Build bottom-up, layer by layer
* Slowly increase intelligence
* Do not simulate humans
* Keep experimentation open-ended
* Embrace unpredictability (agents should surprise us)

The name “Half_Stupid” reflects the idea that the agents begin not intelligent but capable of becoming something interesting through evolution and environment.

---

## **Current Status**

* Malmo environment configured
* Minecraft with Malmo mod working
* Local project directory with an internal conda environment
* MalmoPython bindings accessible (via PYTHONPATH or copying the `.pyd`)
* Basic agent test scripts working
* Initial world design (Paradise + Temple + Maze + Exit structure) drafted
* Cognitive architecture outline prepared
* Implementation phase comes next

---

## **Next Steps**

1. Implement a minimal agent skeleton:
   * vision → action loop
   * short-term memory
   * curiosity-driven exploration
2. Build Paradise v1
3. Implement Temple + Maze mission
4. Add reproduction + generations
5. Add simple valence/emotion system
6. Start experiments on emergent behavior

---

## **How to use this file**

Every time you start working on the project:

* Read this file to refresh the high-level design
* Keep it updated when direction changes
* Do not use it for low-level notes (that goes in `docs/`)
* AI assistants should read this file before generating code or ideas
