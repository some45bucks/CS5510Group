import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import Problem2 as Problem2

def save_lists_to_csv(filename, **kwargs):
    """
    Save lists to a CSV file with corresponding labels.

    Parameters:
    filename (str): Name of the file to save the data
    kwargs (dict): Arbitrary number of lists with corresponding labels
    """
    # Check if the lengths of all lists are the same
    lengths = [len(lst) for lst in kwargs.values()]
    if len(set(lengths)) != 1:
        raise ValueError("All lists must have the same length")

    # Open the file in write mode
    with open(filename, mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)

        # Write the header
        writer.writerow(kwargs.keys())

        # Write the data points
        for data_points in zip(*kwargs.values()):
            writer.writerow(data_points)

def Q2A(depth = 100):

    x,y,t,a,d = Problem2.solve2A(depth)

    save_lists_to_csv("./Question2/Q2A/Q2A_Path_Data.csv",X=x,Y=y,Theta=t,Action=a,Distance=d)

def Q2B(depth = 100):

    x,y,t,al,ar,d = Problem2.solve2B(depth)

    save_lists_to_csv("./Question2/Q2B/Q2B_Path_Data.csv",X=x,Y=y,Theta=t,ActionL=al,ActionR=ar,Distance=d)

def Q2C():

    x, y, t, rx ,ry, e1, t1 = Problem2.solve2C("t1")

    save_lists_to_csv(f"./Question2/Q2C/t1/Q2C_Best_Path_Data_t1.csv",X=x,Y=y,Theta=t,RealX=rx,RealY=ry)

    x, y, t, rx ,ry, e2, t2 = Problem2.solve2C("t_1")

    save_lists_to_csv(f"./Question2/Q2C/t_1/Q2C_Best_Path_Data_t_1.csv",X=x,Y=y,Theta=t,RealX=rx,RealY=ry)

    x, y, t, rx ,ry, e3, t3 = Problem2.solve2C("t_01")

    save_lists_to_csv(f"./Question2/Q2C/t_01/Q2C_Best_Path_Data_t_01.csv",X=x,Y=y,Theta=t,RealX=rx,RealY=ry)

    fig, ax = plt.subplots()
    plt.title("Error/steps and Compute Time over 100 iterations")
    plt.xlabel("Compute Time")
    plt.ylabel("Error/steps")
    custom_lines = [Line2D([0], [0], color='blue', lw=4),
            Line2D([0], [0], color='orange', lw=4),
            Line2D([0], [0], color='green', lw=4),]
    
    ax.legend(custom_lines, ['t = 1', 't = .1', 't = .01'])
    ax.plot(t1, e1, '*')
    ax.plot(t2, e2, '*')
    ax.plot(t3, e3, '*')
    fig.savefig(f"./Question2/Q2C/Q2C_Error.png")
    plt.close()

def Q2D():

    x, y, t, rx ,ry, e1, t1 = Problem2.solve2D("t1")

    save_lists_to_csv(f"./Question2/Q2D/t1/Q2D_Best_Path_Data_t1.csv",X=x,Y=y,Theta=t,RealX=rx,RealY=ry)

    x, y, t, rx ,ry, e2, t2 = Problem2.solve2D("t_1")

    save_lists_to_csv(f"./Question2/Q2D/t_1/Q2D_Best_Path_Data_t_1.csv",X=x,Y=y,Theta=t,RealX=rx,RealY=ry)

    x, y, t, rx ,ry, e3, t3 = Problem2.solve2D("t_01")

    save_lists_to_csv(f"./Question2/Q2D/t_01/Q2D_Best_Path_Data_t_01.csv",X=x,Y=y,Theta=t,RealX=rx,RealY=ry)

    fig, ax = plt.subplots()
    plt.title("Error/steps and Compute Time over 100 iterations")
    plt.xlabel("Compute Time")
    plt.ylabel("Error/steps")
    custom_lines = [Line2D([0], [0], color='blue', lw=4),
            Line2D([0], [0], color='orange', lw=4),
            Line2D([0], [0], color='green', lw=4),]
    
    ax.legend(custom_lines, ['t = 1', 't = .1', 't = .01'])
    ax.plot(t1, e1, '*')
    ax.plot(t2, e2, '*')
    ax.plot(t3, e3, '*')
    fig.savefig(f"./Question2/Q2D/Q2D_Error.png")
    plt.close()


Q2A()
Q2B()
Q2C()
Q2D()