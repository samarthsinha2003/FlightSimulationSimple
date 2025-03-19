# Flight Simulation Algorithm Prototype

This project provides a simple Python prototype for planning an aircraft's flight route over a 100×100 grid, where the goal is to cover at least 80% of the free cells using a 2×3 sensor footprint. The aircraft can only move forward, turn left, or turn right (no backward moves), and obstacles are placed randomly on the grid.

## Overview

- **Grid Representation:** The grid is modeled as a 100×100 NumPy array, where free cells are `0` and obstacles are `1`.
- **Movement:** The aircraft starts at a designated starting position with an initial orientation. At each step, it considers moving forward, turning left, or turning right.
- **Sensor Footprint:** When the aircraft moves, a 2×3 rectangle of cells in front of the aircraft is marked as "covered."
- **Coverage Goal:** The simulation continues until at least 80% of the free cells have been covered by the sensor.
- **Visualization:** Matplotlib is used to visualize the grid, obstacles, sensor coverage, and the flight path.

> **Note:** This prototype demonstrates an example approach (a greedy algorithm) for flight path planning and is not a full simulation solution.

- **Path (Red Line):** The aircraft’s route seems to navigate around obstacles (black squares) without collisions.

- **Coverage (Light Blue):** A large area of free cells is shaded light blue, indicating they have been “seen” by the 2×3 sensor.

- **Stopping Condition:** The provided code stops either when coverage ≥ 80% or when it reaches the maximum step limit. If the run ended because it hit 80% coverage (rather than max steps), then it accomplished the task.

## Getting Started

### Prerequisites

Make sure you have Python 3 installed. Then, install the required libraries using:

```bash
pip install -r requirements.txt
