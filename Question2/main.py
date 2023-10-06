import csv

import Problem2 as Problem2

def save_points_to_file(x_points, y_points,thetas,actions, filename):

    # Check if the lengths of x and y points are the same
    if len(x_points) != len(y_points):
        raise ValueError("x_points and y_points must have the same length")

    # Open the file in write mode
    with open(filename, mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)

        # Write the header
        writer.writerow(['X Pos', 'Y Pos','Current Theta','Action (turn rad)'])

        # Write the data points
        for x, y,t, a in zip(x_points, y_points, thetas,actions):
            writer.writerow([x, y,t, a])

def Q2A():
    Problem2.train2A(50000,100)

    x,y,t,a = Problem2.test2A(True,True,False)

    save_points_to_file(x,y,t,a,"cardata1.csv")

