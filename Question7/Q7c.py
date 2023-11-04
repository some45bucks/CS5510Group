import matplotlib.pyplot as plt
import math
from PlanningAlgorithms.planner import planner  # Import your A* planner class
from hybrid_a_star import hybrid_a_star_planning, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION, WB
from ss_car import LF, LB, get_wheel_speeds
import numpy as np

show_animation = True

def generate_map(obstacle_list, grid_size, robot_radius):
    # Determine the dimensions (rows and columns) of the grid
    max_x = int(max(obstacle[0] + obstacle[2] for obstacle in obstacle_list))
    max_y = int(max(obstacle[1] + obstacle[2] for obstacle in obstacle_list))

    # Calculate the number of rows and columns based on the grid size
    num_rows = int(max_y / grid_size)+1
    num_cols = int(max_x / grid_size)+1

    # Create a flipped map with reversed rows
    map = [[0] * num_cols for _ in range(num_rows)][::-1]

    # Fill the grid with obstacles based on obstacle positions
    for obstacle in obstacle_list:
        x, y, radius = obstacle
        # Calculate the grid cell that contains the center of the obstacle
        row = num_rows - 1 - int(y / grid_size)  # Flip the row calculation
        col = int(x / grid_size)

        # Mark grid cells as obstacles
        map[row][col] = 1

    return map



# Define the start and goal positions
sx, sy = 20.0, 20.0
gx, gy = 60.0, 60.0

# Set obstacle positions
ox, oy = [], []
for i in range(0, 70):
    ox.append(i)
    oy.append(0.0)
for i in range(0, 70):
    ox.append(70.0)
    oy.append(i)
for i in range(0, 71):
    ox.append(i)
    oy.append(70.0)
for i in range(0, 71):
    ox.append(0.0)
    oy.append(i)
for i in range(0, 50):
    ox.append(30.0)
    oy.append(i)
for i in range(0, 50):
    ox.append(50.0)
    oy.append(70.0 - i)
    
# Grid settings
grid_size = 2
robot_radius = LF + LB
obstacle_list = []

for i in range(len(ox)):
    obstacle_list.append([int(ox[i]), int(oy[i]), 1])
    

# Run the A* algorithm
map = generate_map(obstacle_list, 1, robot_radius)  
start = (50, 20)
end = (10, 60)
implemented_planner_parameters = [map, start, end, True, math.ceil(robot_radius)]
a_star_path = planner(implemented_planner_parameters[0], implemented_planner_parameters[1], implemented_planner_parameters[2], implemented_planner_parameters[3], implemented_planner_parameters[4])


def convert_path_to_coordinates(path, grid_height):
    plot_coordinates = []

    for coordinate in path:
        # Assuming coordinate is a tuple (x, y)
        x, y = coordinate
        # Adjust the y-coordinate to match the top-left corner as (0, 0)
        adjusted_x = grid_height - x
        plot_coordinates.append((y, adjusted_x))

    return plot_coordinates



a_star_path=convert_path_to_coordinates(a_star_path, 70)

rx_astar = [x[0] for x in a_star_path]
ry_astar = [x[1] for x in a_star_path]


# Run the Hybrid A* algorithm
path = hybrid_a_star_planning(
    [sx, sy, np.deg2rad(90.0)], [gx, gy, np.deg2rad(-90.0)], ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)

rx_hybrid, ry_hybrid = path.x_list, path.y_list

# Calculate path lengths
length_astar = sum(math.hypot(dx, dy) for dx, dy in zip(np.diff(rx_astar), np.diff(ry_astar)))
length_hybrid = sum(math.hypot(dx, dy) for dx, dy in zip(np.diff(rx_hybrid), np.diff(ry_hybrid)))

wheel_speeds = get_wheel_speeds()

# Plot the paths
plt.figure(figsize=(10, 8))
plt.plot(ox, oy, "s", color='gray', label="Obstacles")
plt.plot(sx, sy, "o", label="Start")
plt.plot(gx, gy, "xg", label="Goal")
plt.plot(rx_astar, ry_astar, "-r", label=f"A* Path (Length: {length_astar:.2f} m)")
plt.plot(rx_hybrid, ry_hybrid, "--b", label=f"Hybrid A* Path (Length: {length_hybrid:.2f} m)")

# Add labels next to the path for every 20 coordinates
for i, (x, y) in enumerate(zip(rx_hybrid, ry_hybrid)):
    left_wheel_speed = wheel_speeds[i][0]
    right_wheel_speed = wheel_speeds[i][1]
    if i == 240 or i == 1250:
        plt.plot(x, y, marker='^', markersize=8, color='c')
        plt.text(x, y, f'(Left: {left_wheel_speed:.2f}, Right: {right_wheel_speed:.2f})', fontsize=8, ha='center', fontweight='bold')

plt.legend()
plt.grid(True)
plt.axis("equal")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Comparison of A* and Hybrid A* Paths")
plt.savefig('./Question7/Q7c_implemented_hybrid_Astar.png')

# Display the graph
if show_animation:
    plt.show()
