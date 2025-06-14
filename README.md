# Two-Watchtowers Visibility Problem

## Overview

This repository contains Python implementations of algorithms designed to solve the **Two-Watchtowers Problem** over an x-monotone polygonal terrain. The objective is to place two vertical towers of minimum height such that every point on the terrain is visible from at least one tower.

The algorithms support:

* **Discrete Mode**: Both towers are placed on terrain vertices.
* **Semi-Continuous Mode**: One tower is on a vertex, the second can be on a vertex or inside an edge.
* **Discrete Mode without Parametric Search**: Exhaustive binary search over critical heights.

These are based on the thesis work by *Panagiotis Mouzouris* at the University of Ioannina, 2025.

---

## File Structure

* `Discrete.py`
  Implements the **discrete version** using a parametric search over the upper envelope formed by non-visible terrain segments.

* `Semi_Continues.py`
  Extends the previous algorithm for the **semi-continuous case**, allowing the second tower to be placed anywhere along the terrain.

* `Discrete_Avoid_Parametric.py`
  Solves the discrete version **without parametric search**, using SPT (Shortest Path Trees), type-1 and type-2 critical heights, and binary search over them.

* `mouzourhs_keimeno.pdf`
  The thesis document in Greek, describing the theoretical foundations, algorithmic details, and implementation rationale.

---

## How It Works

1. **Non-Visible Chain Detection**
   Identify terrain segments not visible from the first tower using cross products and triangle area checks.

2. **Upper Envelope Construction**
   Compute the upper envelope of the non-visible segments (transformed into lines), using duality and Graham Scan convex hull.

3. **Height Optimization**
   Find the minimum additional height (`h2`) required for the second tower to ensure full visibility of the terrain.

---

## Usage

Each script can be modified to define:

* `terrain`: a list of `(x, y)` coordinates forming the x-monotone polygonal chain.
* `u_idx`: index of the vertex for the first tower.
* `h1`: height of the first tower.

For example:

```python
terrain = [(0, 0), (1, 3), (2, 0), (3, 3), (4, 0), (5, 8), (7, 0)]
u_idx = 0
h1 = 0
```

Then call the relevant method to run the search and display or print results.

---

## Dependencies

All scripts use only the following Python libraries:

* `math`
* `numpy`
* `matplotlib`
* `typing`
* `time`

Install requirements (if needed):

```bash
pip install numpy matplotlib
```

---

## Example Output

For `Discrete.py`, the output might look like:

```
Minimum h2: 3.0
Best location for second tower: (7, 0)
```

For `Semi_Continues.py`, it can output:

```
Minimum h2: 0.99
Best location for second tower: (7.66, 5.66)
```

---

## License

This project is part of an academic thesis and is free to use for educational or research purposes. Please cite appropriately if used.

---

