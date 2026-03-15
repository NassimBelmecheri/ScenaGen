# ScenaGen CLI – Qualitative Scenario Generator and Solver

## Overview

**ScenaGen CLI** is a command-line tool for generating and solving qualitative spatial–temporal scenarios involving traffic participants such as cars, pedestrians, buses, trucks, and an ego vehicle.

The system models relationships between objects using:

* **Allen interval algebra** for spatial relations (RA)
* **Qualitative Distance Calculus (QDC)** for proximity
* **Temporal motion constraints** based on speed limits

The solver reconstructs object positions across multiple frames using **constraint programming with MiniZinc**.

The tool supports:

* random scenario generation
* importing qualitative relations from CSV datasets
* solving spatial–temporal constraints
* visualization of scenarios
* exporting results and solver statistics

---

## Features

### Scenario Generation

Generate random traffic scenarios with configurable:

* number of objects
* number of frames
* object categories
* object headings
* speed categories

Supported object types:

* ego vehicle
* car
* pedestrian
* bus
* truck

Relations between objects are automatically computed.

---

### Constraint Solver

The solver reconstructs spatial configurations satisfying:

* Allen spatial relations
* qualitative distance constraints
* speed limits between frames
* object size hierarchy
* ego vehicle fixed at origin

The constraint model is built dynamically and solved using **MiniZinc**.

---

### Progressive Constraint Refinement

The solver can run using **incremental constraint refinement**:

1. Start with a reduced constraint set
2. Gradually reintroduce constraints
3. Use previous solutions as guidance

This allows analysis of solver behavior under different constraint densities.

---

### Dataset Import

Scenarios can be loaded from CSV datasets containing qualitative relations.

The loader extracts:

* object identities
* qualitative relations (RA)
* qualitative distances (QDC)
* object headings
* speed categories
* frame indices

The system then reconstructs the qualitative scenario graph.

---

### Visualization

Scenarios can be visualized using **Matplotlib**.

Features include:

* per-frame visualizations
* bounding boxes for objects
* heading indicators
* color-coded object categories
* ego-centered coordinate system

Plots can be displayed interactively or exported as images.

---

## Architecture

### Config

Defines system parameters including:

* object dimensions
* map size calculation
* speed limits
* qualitative distance thresholds
* visualization colors

---

### GlobalScenarioSolver

Handles constraint solving.

Responsibilities include:

* building the MiniZinc model
* encoding qualitative constraints
* solving the constraint problem
* extracting spatial solutions

---

### CLIScenarioDesigner

High-level interface responsible for:

* generating scenarios
* loading datasets
* executing the solver
* visualization
* exporting results

---

## Qualitative Relations

### Allen Spatial Relations (RA)

Relations applied to bounding boxes along the X and Y axes.

Examples:

* Before
* After
* Meets
* Overlaps
* During
* Starts
* Finishes
* Equals

---

### Qualitative Distance Calculus (QDC)

Distance categories between objects:

* very close
* close
* normal
* far
* very far

Distances are computed from bounding box separation.

---

## Installation

### Requirements

Python 3.9+

Install Python dependencies:

```
pip install pandas numpy matplotlib nest_asyncio minizinc
```

Install MiniZinc from:

https://www.minizinc.org/software.html

Make sure a solver such as **Gecode** is installed.

---

## Usage

### Generate a Random Scenario

```
from scenagen_cli import CLIScenarioDesigner

designer = CLIScenarioDesigner()

designer.generate_random_scenario(
    num_objects=5,
    num_frames=3,
    include_ego=True,
    seed=42
)
```

---

### Solve the Scenario

```
results, stats = designer.solve(
    solver_name="gecode",
    heuristic="default",
    timeout=60,
    refinements=5
)
```

---

### Plot the Scenario

```
designer.plot_scenario(results[-1])
```

---

### Export Solution

```
designer.export_to_csv(results[-1], "solution.csv")
```

---

### Save Qualitative Relations

```
designer.save_qualitative_relations("relations.csv")
```

---

## Solver Statistics

The system records solver metrics including:

* solving time
* first solution time
* number of nodes
* number of failures
* propagations
* memory usage

Statistics can be exported for experimental analysis.

---

## Example Workflow

Typical pipeline:

1. Generate or import a scenario
2. Compute qualitative relations
3. Run the constraint solver
4. Visualize results
5. Export solver statistics

---


