import math
import matplotlib.pyplot as plt

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

# Visualization
plt.plot(ox, oy, 'sk', label='Obstacles')
plt.plot(sx, sy, 'og', label='Start')
plt.plot(gx, gy, 'or', label='Goal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Obstacle Field')
plt.legend()

# Implement a simplified Hybrid A* algorithm
def hybrid_a_star(start, goal, obstacle_x, obstacle_y):
    # Define the Hybrid A* algorithm for path planning
    def heuristic(node, goal):
        # Implement your heuristic function (e.g., Euclidean distance)
        x1, y1 = node
        x2, y2 = goal
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def is_valid_state(node, obstacle_x, obstacle_y):
        # Implement collision checking here
        return all(
            not ((ox - node[0]) ** 2 + (oy - node[1]) ** 2 < 1)
            for (ox, oy) in zip(obstacle_x, obstacle_y)
        )

    # Add your Hybrid A* logic here
    # ...

    return None  # No valid path found

# Call the simplified Hybrid A* algorithm
path = hybrid_a_star((sx, sy), (gx, gy), ox, oy)

if path is not None:
    # Extract x and y coordinates from the path
    x_path, y_path = zip(*path)
    plt.plot(x_path, y_path, '-b', label='Hybrid A* Path')
    plt.legend()
    plt.show()
else:
    print("No valid path found.")
