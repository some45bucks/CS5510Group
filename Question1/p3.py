import numpy as np

g = 9.81  # Acceleration due to gravity
m_c = 3.0  # Mass of the cart (M)
m_p = 0.5  # Mass of the pendulum (m)
l = 0.8  # Length of the pendulum (l)
max_force = 7.0  # Maximum applied force

theta_vel = 0.0 # Assume there was no initial angular velocity
theta_acc = 0.0 # Assume there was no initial angular acceleration
x_vel = 0.0 # Assume there was no initial linear velocity
x_acc = 0.0 # Assume there was no initial linear acceleration

# helpful calculations
masspole_length = m_p * l
total_mass = m_p + m_c

# Iterate through different initial angles from 0.01 to pi/2 radians
for theta in np.linspace(0.01, np.pi / 2, 1000):
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    masspole_length = m_p * l
    total_mass = m_p + m_c
    
    temp = (
        max_force + masspole_length * theta_vel ** 2 * sintheta
    ) / total_mass
    theta_acc = (g * sintheta - costheta * temp) / (
        l * (4.0 / 3.0 - m_p * costheta ** 2 / total_mass)
    )
    x_acc = temp - masspole_length * theta_acc * costheta / total_mass

    # If the angular acceleration becomes positive it is increasing thus, falling.
    if theta_acc > 0:
        print(f"Critical Angle (Radians): {theta}, Angular Acceleration: {theta_acc}")
        break
    else:
        print(f"Angle (Radians): {theta}, Angular Acceleration: {theta_acc}")