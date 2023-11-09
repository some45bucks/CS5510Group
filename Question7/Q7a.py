from PlanningAlgorithms import a_star, bd_a_star, bfs, dijkstra, rrt_star, planner

import math
import time
import random
import pandas as pd
import matplotlib.pyplot as plt

def generate_map(obstacle_list, robot_radius):
    # Determine the dimensions (rows and columns) of the grid
    max_x = int(max(obstacle[0] + obstacle[2] for obstacle in obstacle_list) + robot_radius)
    max_y = int(max(obstacle[1] + obstacle[2] for obstacle in obstacle_list) + robot_radius)

    # Calculate the number of rows and columns based on the grid size
    num_rows = int(max_y)+1
    num_cols = int(max_x)+1

    map = [[0] * num_cols for _ in range(num_rows)]

    # Fill the grid with obstacles based on obstacle positions
    for obstacle in obstacle_list:
        x, y, radius = obstacle
        # Calculate the grid cell that contains the center of the obstacle
        row = int(y / 1)
        col = int(x / 1)

        # Mark grid cells as obstacles
        map[row][col] = 1

    return map

def run_implemented_planner(parameters, iterations=10):
    total_cost = 0
    total_time = 0
    for _ in range(iterations):
        start_time = time.time()
        path = planner.planner(parameters[0], parameters[1], parameters[2], parameters[3])
        grid_path = []
        for x, y in path:
            row = int(y)
            col = int(x)
            grid_path.append((row, col))

        end_time = time.time()
        planning_time = end_time - start_time
        total_time += planning_time

        # Calculate the cost of the path (e.g., total distance)
        path_cost = 0
        for i in range(1, len(path)):
            x1, y1 = path[i - 1]
            x2, y2 = path[i]
            path_cost += math.hypot(x2 - x1, y2 - y1)
        
        total_cost += path_cost

    average_cost = total_cost / iterations
    average_time = total_time * 1000 / iterations

    return average_cost, average_time

def run_planner(planner, start, goal, iterations=10):
    total_cost = 0
    total_time = 0
    for _ in range(iterations):
        start_time = time.time()
        rx, ry = planner.planning(start[0], start[1], goal[0], goal[1])
        end_time = time.time()
        planning_time = end_time - start_time
        total_time += planning_time

        # Calculate the cost of the path (e.g., total distance)
        path_cost = sum(math.hypot(rx[i] - rx[i - 1], ry[i] - ry[i - 1]) for i in range(1, len(rx)))

        total_cost += path_cost

    average_cost = total_cost / iterations
    average_time = total_time * 1000 / iterations

    return average_cost, average_time

def run_rtt_planner(planner, iterations=10):
    total_cost = 0
    total_time = 0
    for _ in range(iterations):
        start_time = time.time()
        path = planner.planning(animation=False)
        end_time = time.time()
        planning_time = end_time - start_time
        total_time += planning_time

        # Calculate the cost of the path (e.g., total distance)
        path_cost = sum(math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) for i in range(1, len(path)))

        total_cost += path_cost

    average_cost = total_cost / iterations
    average_time = total_time * 1000 / iterations

    return average_cost, average_time

if __name__ == '__main__':
    # Define your grid map parameters, start, and goal positions
    sx = 10.0
    sy = 10.0
    gx = 50.0
    gy = 50.0
    grid_size = 2.0
    robot_radius = 1.0

    random.seed(1)

    # Define obstacle positions (ox, oy)
    num_obstacles = 50
    obstacle_list = []

    while len(obstacle_list) < num_obstacles:
        x = random.uniform(0, 60)
        y = random.uniform(0, 60)
        radius = 1.0  # Set the radius of the obstacle

        overlap = False
        for obstacle in obstacle_list:
            existing_x, existing_y, existing_radius = obstacle
            distance = math.sqrt((x - existing_x) ** 2 + (y - existing_y) ** 2)
            if distance < radius + existing_radius:
                overlap = True
                break

        if not overlap:
            obstacle_list.append((x, y, radius))
            
    ox = [obstacle[0] for obstacle in obstacle_list]
    oy = [obstacle[1] for obstacle in obstacle_list]
    
    map = generate_map(obstacle_list, robot_radius)    
    # Create a map image
    plt.figure(figsize=(8, 8))
    plt.plot(ox, oy, "ok", markersize=10)  # Obstacle positions
    plt.plot(sx, sy, "go", markersize=10)  # Start position
    plt.plot(gx, gy, "bo", markersize=10)  # Goal position

    plt.grid(True)
    plt.axis("equal")
    plt.title("Grid Map with Obstacles")
    plt.show()

    start = (int(sx), int(sy))
    end = (int(gx), int(gy))
    implemented_planner_parameters = [map, start, end, True]
    a_star_planner = a_star.AStarPlanner(ox, oy, grid_size, robot_radius)
    bidir_a_star_planner = bd_a_star.BidirectionalAStarPlanner(ox, oy, grid_size, robot_radius)
    dijkstra_planner = dijkstra.Dijkstra(ox, oy, grid_size, robot_radius)
    bfs_planner = bfs.BreadthFirstSearchPlanner(ox, oy, grid_size, robot_radius)
    rrt_star_planner = rrt_star.RRTStar(
        start=[sx, sy],
        goal=[gx, gy],
        obstacle_list=obstacle_list,
        rand_area=[0, 60, 0, 60],
        robot_radius=robot_radius
    )

    # Run the planner for 10 iterations and calculate average cost and time
    implemented_avg_cost, implemented_avg_time = run_implemented_planner(implemented_planner_parameters)
    a_star_avg_cost, a_star_avg_time = run_planner(a_star_planner, (sx, sy), (gx, gy))
    bidir_a_star_avg_cost, bidir_a_star_avg_time = run_planner(bidir_a_star_planner, (sx, sy), (gx, gy))
    dijkstra_avg_cost, dijkstra_avg_time = run_planner(dijkstra_planner, (sx, sy), (gx, gy))
    bfs_avg_cost, bfs_avg_time = run_planner(bfs_planner, (sx, sy), (gx, gy))
    rrt_star_avg_cost, rrt_star_avg_time = run_rtt_planner(rrt_star_planner)

    # Create a table
    data = {
        'Algorithm': ['Implemented A*', 'A*', 'Bidirectional A*', 'Breadth First Search', 'Dijkstra', 'RRT*'],
        'Average Path Cost': [implemented_avg_cost, a_star_avg_cost, bidir_a_star_avg_cost, bfs_avg_cost, dijkstra_avg_cost, rrt_star_avg_cost],
        'Average Time (ms)': [implemented_avg_time, a_star_avg_time, bidir_a_star_avg_time, bfs_avg_time, dijkstra_avg_time, rrt_star_avg_time]
    }

    table = pd.DataFrame(data)
    print(table)
