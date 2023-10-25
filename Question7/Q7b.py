from queue import PriorityQueue
import matplotlib.pyplot as plt
import numpy as np
import math 
# Define the terrain grid
grid = [
    [4, 3, 2, 2, 2, 2, 1, 1, 1, 1],
    [3, 2, 2, float('inf'), 2, 1, 1, 1, 2, 1],
    [3, 2, 2, float('inf'), 2, 1, 't', 3, 3, 1],
    [2, 2, 1, float('inf'), 4, 2, 4, 4, 3, 1],
    [2, 1, 1, float('inf'), 4, 4, 4, 4, 2, 1],
    [1, 1, 1, float('inf'), 3, 4, 3, 2, 2, 1],
    [1, 2, 2, 2, 2, 3, 1, 1, 1, 1],
    ['s', 1, 1, 1, 1, 1, 1, 1, 2, 2]
]

# Define coordinate directions for neighbors (up, down, left, right, diagonals)
dx = [-1, 1, 0, 0, -1, -1, 1, 1]
dy = [0, 0, -1, 1, -1, 1, -1, 1]

# Define functions as per the provided pseudocode
def aug_dist(u, v, display=False):
    if display:
        print(f'{u} -> {v}', end=' = ')
    if u == 's' or u == 't':
        u = 1
    elif v == 't' or v == 's':
        v = 1
    if u < v:
        if display:
            print(f'{2}')
        return 2
    elif u > v:
        if display:
            print(f'{0.5}')
        return 0.5
    if display:
        print(f'{1}')
    return 1

def heuristic(u, t):
    return math.sqrt((u[0] - t[0])**2 + (u[1] - t[1])**2)

# Define A* search algorithm
def a_star_search(grid, start, target):
    n, m = len(grid), len(grid[0])
    q = PriorityQueue()
    q.put((0, start))
    exp = set()
    parent = {}
    cost = {(i, j): float('inf') for i in range(n) for j in range(m)}
    cost[start] = 0

    while not q.empty():
        _, u = q.get()
        if u == target:
            path = [u]
            while u in parent:
                u = parent[u]
                path.append(u)
            return path[::-1]

        exp.add(u)

        for i in range(8):
            x, y = u[0] + dx[i], u[1] + dy[i]
            if 0 <= x < n and 0 <= y < m and grid[x][y] != float('inf'):
                v = (x, y)
                work = cost[u] + aug_dist(grid[u[0]][u[1]], grid[x][y])
                if work < cost[v]:
                    cost[v] = work
                    priority = work + heuristic(v, target)
                    q.put((priority, v))
                    parent[v] = u

    return None

# Find the start and target positions
start = None
target = None
for i in range(len(grid)):
    for j in range(len(grid[0])):
        if grid[i][j] == 's':
            start = (i, j)
        elif grid[i][j] == 't':
            target = (i, j)

# Run A* search
if start and target:
    path = a_star_search(grid, start, target)
    if path:
        total_cost = 0
        print("Path found:")
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            print(u, '->', v, end=': ')
            cost = aug_dist(grid[u[0]][u[1]], grid[v[0]][v[1]], display=True)
            total_cost += cost
        print("Total Cost of the Path: {:.2f}".format(total_cost))
    else:
        print("No path found.")
else:
    print("Start and/or target positions not found in the grid.")

# Define custom colors for each key in the 'colors' dictionary using RGB values
custom_colors = {
    float('inf'): (0, 0, 255),        # Black for float('inf')
    4: (240, 5, 5),                 # Blue for 4
    3: (253, 97, 4),                 # Green for 3
    2: (253, 154, 1),               # Yellow for 2
    1: (254, 240, 1),                 # Red for 1
    't': (255, 255, 255),             # Purple for 't'
    's': (255, 255, 255)              # Orange for 's'
}

# Convert the grid to an array for visualization using custom colors
grid_array = np.array([[custom_colors[cell] for cell in row] for row in grid])

# Create a custom colormap using ListedColormap for discrete colors
cmap = plt.matplotlib.colors.ListedColormap(list(custom_colors.values()))

# Extract x and y coordinates of the path
path_x, path_y = zip(*path)

# Visualize the grid with the custom colormap
plt.imshow(grid_array, cmap=cmap, interpolation='none')
0
# Highlight the path
plt.plot(path_y, path_x, marker='o', color='green', markersize=8)

# Add labels for start and target positions
for i in range(len(grid)):
    for j in range(len(grid[0])):
        if grid[i][j] == 's':
            plt.text(j, i, 'S', fontsize=12, ha='center', va='center', color='black')
        elif grid[i][j] == 't':
            plt.text(j, i, 'T', fontsize=12, ha='center', va='center', color='black')

# Set axis labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('A* Search Path Visualization')

# Show the grid and path
plt.savefig('./Question7/Q7b_optimal_path_found.png')
plt.show()