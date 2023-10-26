import matplotlib.pyplot as plt
import math
from PlanningAlgorithms.a_star import AStarPlanner  # Import your A* planner class
from hybrid_a_star import hybrid_a_star_planning, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION, WB
from car import LF, LB
import numpy as np

show_animation = True

# Define the start and goal positions
sx, sy = 10.0, 10.0
gx, gy = 50.0, 50.0

# Set obstacle positions
ox, oy = [], []
for i in range(-10, 60):
    ox.append(i)
    oy.append(-10.0)
for i in range(-10, 60):
    ox.append(60.0)
    oy.append(i)
for i in range(-10, 61):
    ox.append(i)
    oy.append(60.0)
for i in range(-10, 61):
    ox.append(-10.0)
    oy.append(i)
for i in range(-10, 40):
    ox.append(20.0)
    oy.append(i)
for i in range(0, 40):
    ox.append(40.0)
    oy.append(60.0 - i)
    
# Grid settings
grid_size = 2
robot_radius = LF + LB

# Run the A* algorithm
a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
rx_astar, ry_astar = a_star.planning(sx, sy, gx, gy)

# Run the Hybrid A* algorithm
path = hybrid_a_star_planning(
    [sx, sy, np.deg2rad(90.0)], [gx, gy, np.deg2rad(-90.0)], ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)

rx_hybrid, ry_hybrid = path.x_list, path.y_list

# Calculate path lengths
length_astar = sum(math.hypot(dx, dy) for dx, dy in zip(np.diff(rx_astar), np.diff(ry_astar)))
length_hybrid = sum(math.hypot(dx, dy) for dx, dy in zip(np.diff(rx_hybrid), np.diff(ry_hybrid)))

# Plot the paths
plt.figure(figsize=(10, 8))
plt.plot(ox, oy, ".k", label="Obstacles")
plt.plot(sx, sy, "og", label="Start")
plt.plot(gx, gy, "xb", label="Goal")
plt.plot(rx_astar, ry_astar, "-r", label=f"A* Path (Length: {length_astar:.2f} m)")
plt.plot(rx_hybrid, ry_hybrid, "-b", label=f"Hybrid A* Path (Length: {length_hybrid:.2f} m)")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Comparison of A* and Hybrid A* Paths")

# Display the graph
if show_animation:
    plt.show()
