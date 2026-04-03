# ASTR 311 Project Planning Document

## Gravitational Simulation of Solar System Formation

---

## 1. Project Overview

Our project focuses on building a visual simulation that demonstrates how gravity can organize matter into stable orbital structures over long periods of time. We will model a simplified gravitational system in which a large number of small particles interact with a central mass (representing a star), allowing us to observe how structure can emerge through gravitational attraction alone.

The goal is not to reproduce realistic astrophysical planet formation in full detail, but to create a simplified, educational model that helps us understand how gravity shapes motion and structure in space.

The final result will be presented as an interactive 3D visualization that allows the class to observe how systems evolve over time and how changing certain parameters affects outcomes.

---

## 2. Core Scientific Concepts

This project builds directly on course concepts:

* Newtonian gravitational force
* Orbital motion and stability
* Multi-body gravitational interactions
* Emergent structure from many interacting masses
* The role of dominant central mass in system organization

By implementing these principles computationally, we aim to better understand:

* How stable orbits emerge from initial conditions
* How particle distribution influences long-term evolution
* How gravitational systems can self-organize
* The limitations of simplified N-body simulations compared to real astrophysical systems

---

## 3. Model Scope and Assumptions

To keep the model focused and manageable:

* Gravity will be modeled using **Newtonian mechanics**
* Particles will be treated as point masses
* No gas dynamics, magnetic fields, or radiation effects
* Gravitational softening will be used to prevent numerical instability
* The simulation emphasizes visual clarity and conceptual understanding

We will explicitly discuss how these simplifications differ from real planetary formation processes.

---

## 4. Computational Plan

The simulation will be developed and executed on a dedicated virtual machine equipped with:

* 24 logical CPU cores
* 31 GB RAM
* NVIDIA RTX A2000 (12GB VRAM, CUDA-enabled)

### Phase 1: 2D Prototype (CPU-Based)

* Implement gravity and numerical integration in 2D.
* Validate stability of orbital motion.
* Ensure conservation behavior (energy trends, bounded motion).
* Test particle disk and cloud initial conditions.

### Phase 2: 3D Extension

* Extend physics to three dimensions.
* Improve visualization and camera control.
* Begin increasing particle count.

### Phase 3: Scaling and Optimization

* Increase particle count toward 100,000+.
* Explore GPU acceleration using CUDA-compatible Python tools if necessary.
* Optimize only after correctness and clarity are established.

### Phase 4: Web-Based Visualization

* Export simulation data (positions over time).
* Build a client-side WebGL viewer for replay and interaction.
* Allow parameter adjustments and visual exploration.

The heavy computation will occur offline; the web component will serve as an interactive replay interface.

---

## 5. Expected Simulation Scale

Given available hardware:

* Early development: 10k–50k particles
* Mid-stage testing: 50k–200k particles
* Stretch goal: approach higher particle counts with optimization

The goal is to balance visual clarity with computational feasibility.

---

## 6. Final Demonstration Goals

The completed project should allow:

* 3D visualization of evolving gravitational systems
* Camera rotation and zoom
* Adjustable initial parameters (particle count, central mass, distribution)
* Replay of system evolution over long timescales
* Class interaction and discussion

---

## 7. Learning Outcomes

Through this project we expect to:

* Develop deeper intuition for gravitational dynamics
* Understand the challenges of modeling many-body systems
* Explore numerical stability and integration methods
* Recognize the strengths and limitations of simplified models
* Improve our ability to communicate physical processes visually

Our presentation will conclude with discussion of:

* What the model successfully demonstrates
* Where it deviates from real astrophysical processes
* How gravity alone contributes to structure formation
* What additional physics would be required for realism

---

## 8. Success Criteria

We consider the project successful if:

* The simulation runs stably over long timescales
* Emergent structure is clearly visible
* The 3D visualization is intuitive and interactive
* We can clearly explain both the physics and the limitations
* The class gains insight from the visual demonstration

---

## 9. VM Hardware

* **GPU:** NVIDIA RTX A2000 12GB
* **CUDA:** 13.1
* **CPU:** Xeon E5-2643 v4 @ 3.40GHz
* **Cores:** 24 logical
* **RAM:** 31 GB
* **OS:** AlmaLinux 10.1