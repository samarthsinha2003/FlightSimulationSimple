import numpy as np
import matplotlib.pyplot as plt

# Grid dimensions
GRID_WIDTH = 100
GRID_HEIGHT = 100

# Coverage threshold (80% of free cells)
COVERAGE_THRESHOLD = 0.8

# Define possible orientations (0: North, 1: East, 2: South, 3: West)
ORIENTATIONS = [0, 1, 2, 3]

# Movement actions: forward (F), left-turn (L), right-turn (R)
ACTIONS = ["F", "L", "R"]

# Map orientations to movement delta for a forward move:
MOVE_DELTA = {
    0: (-1, 0),  # North: decrease row
    1: (0, 1),   # East: increase column
    2: (1, 0),   # South: increase row
    3: (0, -1)   # West: decrease column
}

def create_grid(width, height, obstacle_ratio=0.1):
    """
    Create a grid of given dimensions.
    Cells with value 0 are free, 1 are obstacles.
    obstacle_ratio: fraction of cells that are obstacles.
    """
    grid = np.zeros((height, width), dtype=int)
    num_obstacles = int(obstacle_ratio * width * height)
    obstacle_indices = np.random.choice(width * height, num_obstacles, replace=False)
    grid.flat[obstacle_indices] = 1
    return grid

def get_sensor_footprint(position, orientation):
    """
    Given the aircraft's position (row, col) and orientation, compute the sensor footprint.
    The sensor covers a 2x3 rectangle ahead of the aircraft.
    We assume the sensor rectangle is placed immediately in front of the aircraft.
    """
    row, col = position
    footprint = []
    # Relative coordinates for a 2x3 rectangle with the aircraft at the bottom center when facing North.
    # We'll define it for the North orientation and then rotate for other directions.
    # For North: rectangle is two rows high, three columns wide, in front of the aircraft.
    rel_coords = [(-1, -1), (-1, 0), (-1, 1),
                  (-2, -1), (-2, 0), (-2, 1)]
    
    # Rotate relative coordinates based on orientation
    def rotate(coord, orientation):
        r, c = coord
        # 90 degrees clockwise rotation
        for _ in range(orientation):
            r, c = c, -r
        return (r, c)
    
    for rel in rel_coords:
        dr, dc = rotate(rel, orientation)
        footprint.append((row + dr, col + dc))
    return footprint

def is_valid(position, grid):
    """
    Check if a given position (row, col) is within the grid and not an obstacle.
    """
    row, col = position
    if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
        return grid[row, col] == 0
    return False

def apply_action(position, orientation, action, grid):
    """
    Compute the new position and orientation after applying an action.
    Returns new_position, new_orientation.
    """
    if action == "F":
        # Move forward in current orientation
        dr, dc = MOVE_DELTA[orientation]
        new_position = (position[0] + dr, position[1] + dc)
        new_orientation = orientation
    elif action == "L":
        # Turn left and then move forward
        new_orientation = (orientation - 1) % 4
        dr, dc = MOVE_DELTA[new_orientation]
        new_position = (position[0] + dr, position[1] + dc)
    elif action == "R":
        # Turn right and then move forward
        new_orientation = (orientation + 1) % 4
        dr, dc = MOVE_DELTA[new_orientation]
        new_position = (position[0] + dr, position[1] + dc)
    else:
        raise ValueError("Invalid action")
    
    # Check if the new position is valid; if not, return None.
    if is_valid(new_position, grid):
        return new_position, new_orientation
    else:
        return None, None

def compute_coverage(visited, grid):
    """
    Compute the fraction of free grid cells that have been covered.
    visited is a boolean grid the same size as grid.
    """
    free_cells = np.sum(grid == 0)
    covered_cells = np.sum(visited)
    return covered_cells / free_cells if free_cells > 0 else 0

def simulate_flight(grid, start_position, start_orientation, max_steps=10000):
    """
    Simulate the flight using a greedy strategy:
    At each step, try the three actions and choose the one that maximizes new sensor coverage.
    Stops when coverage threshold is reached or max_steps is exceeded.
    Returns the flight path (list of positions) and orientations.
    """
    current_position = start_position
    current_orientation = start_orientation
    path = [current_position]
    orientations = [current_orientation]
    
    # Create a grid to track sensor-covered cells.
    visited = np.zeros_like(grid, dtype=bool)
    
    # Initially mark sensor footprint as covered.
    for cell in get_sensor_footprint(current_position, current_orientation):
        r, c = cell
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            visited[r, c] = True

    steps = 0
    while compute_coverage(visited, grid) < COVERAGE_THRESHOLD and steps < max_steps:
        best_action = None
        best_new_cells = -1
        best_next_state = (None, None)
        
        # Evaluate each possible action
        for action in ACTIONS:
            next_position, next_orientation = apply_action(current_position, current_orientation, action, grid)
            if next_position is None:
                continue  # invalid move
            
            # Determine sensor coverage from the potential new state.
            footprint = get_sensor_footprint(next_position, next_orientation)
            new_cells = 0
            for (r, c) in footprint:
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    if grid[r, c] == 0 and not visited[r, c]:
                        new_cells += 1
            
            if new_cells > best_new_cells:
                best_new_cells = new_cells
                best_action = action
                best_next_state = (next_position, next_orientation)
        
        # If no action yields new coverage, still choose a valid move (e.g., forward if possible)
        if best_action is None:
            best_next_state = apply_action(current_position, current_orientation, "F", grid)
            if best_next_state[0] is None:
                break  # no valid moves remain
        
        # Update current state
        current_position, current_orientation = best_next_state
        path.append(current_position)
        orientations.append(current_orientation)
        # Update visited cells with new sensor footprint.
        for cell in get_sensor_footprint(current_position, current_orientation):
            r, c = cell
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                visited[r, c] = True
        
        steps += 1

    final_coverage = compute_coverage(visited, grid)
    print(f"Simulation finished after {steps} steps with {final_coverage*100:.2f}% coverage.")
    return path, orientations, visited

def visualize(grid, path, visited):
    """
    Visualize the grid, the flight path, and the sensor coverage.
    Obstacles are in black, free cells in white, visited cells in light blue, and the flight path in red.
    """
    plt.figure(figsize=(8, 8))
    grid_rgb = np.zeros((*grid.shape, 3))
    
    # Color free cells white and obstacles black.
    grid_rgb[grid == 0] = [1, 1, 1]
    grid_rgb[grid == 1] = [0, 0, 0]
    
    # Overlay visited cells (light blue)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if visited[r, c]:
                grid_rgb[r, c] = [0.7, 0.85, 1]
    
    plt.imshow(grid_rgb, origin="upper")
    
    # Overlay the flight path.
    path = np.array(path)
    plt.plot(path[:, 1], path[:, 0], color="red", linewidth=2, marker="o", markersize=3)
    
    plt.title("Flight Path and Sensor Coverage")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.gca().invert_yaxis()
    plt.show()

def main():
    # Create the grid with obstacles
    grid = create_grid(GRID_WIDTH, GRID_HEIGHT, obstacle_ratio=0.1)
    
    # Set starting position and orientation (choose a free cell)
    start_position = (GRID_HEIGHT - 1, 0)  # bottom-left corner
    if grid[start_position] == 1:
        # If starting cell is an obstacle, choose an alternative free cell.
        start_position = (GRID_HEIGHT - 1, 1)
    start_orientation = 0  # Facing North
    
    # Run the simulation
    path, orientations, visited = simulate_flight(grid, start_position, start_orientation)
    
    # Visualize the results
    visualize(grid, path, visited)

if __name__ == "__main__":
    main()
