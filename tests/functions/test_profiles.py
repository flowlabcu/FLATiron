import matplotlib.pyplot as plt
import numpy as np 

from flatiron_tk.functions import ParabolicInletProfile
from flatiron_tk.functions import ParaboloidInletProfile
from flatiron_tk.functions import PlugInletProfile

def test_parabolic():
    num_points = 100 

    center = np.array([0.0, 0.5, 0.0])
    radius = 0.3
    normal = np.array([1.0, 0.0, 0.0])

    u_max = 10 
    Q = u_max * radius * 4 / 3 

    # Sample points along a vertical line through the center
    # Expanding the line to ensure we have points outside the radius (to test zero velocity)
    y_vals = np.linspace(center[1] - (1.2 * radius) , center[1] + (1.2 * radius), num_points)
    x = np.vstack([np.full_like(y_vals, center[0]), y_vals, np.full_like(y_vals, center[2])])
    
    # Manually compute expected values
    # r = |y - center_y|
    r = np.abs(y_vals - center[1])
    factor = np.where(r <= radius, 1 - (r / radius) ** 2, 0.0)
    u_mag_exact = u_max * factor

    excepted_profile = normal[:, None] * u_mag_exact[None, :]
    
    # Create profile and evaluate
    profile = ParabolicInletProfile(Q, radius, center, normal)
    evaluated_profile = profile(x)

    assert np.linalg.norm(evaluated_profile - excepted_profile) < 1e-10

    # Update flow rate and re-test
    u_max = 20
    Q = u_max * radius * 4 / 3
    profile.update_flow_rate(Q)
    evaluated_profile = profile(x)
    u_mag_exact = u_max * factor
    excepted_profile = normal[:, None] * u_mag_exact[None, :]

    assert np.linalg.norm(evaluated_profile - excepted_profile) < 1e-10

def test_parabaloid():
    num_points = 100
    center = np.array([0.0, 0.5, 0.0])
    radius = 0.5
    normal = np.array([0.0, 0.0, 1.0])
    u_max = 10 

    # Hagen-Poiseuille flow rate in a circular pipe
    Q = u_max * np.pi * radius**2 / 2

    # Sample points in a square grid around the center
    x_vals = np.linspace(center[0] - (1.2 * radius) , center[0] + (1.2 * radius), int(np.sqrt(num_points)))
    y_vals = np.linspace(center[1] - (1.2 * radius) , center[1] + (1.2 * radius), int(np.sqrt(num_points)))
    X, Y = np.meshgrid(x_vals, y_vals)
    x = np.vstack([X.ravel(), Y.ravel(), np.full(X.size, center[2])])

    # Manually compute expected values
    r = np.sqrt((x[0, :] - center[0])**2 + (x[1, :] - center[1])**2)
    factor = np.where(r <= radius, 1 - (r / radius) ** 2, 0.0)
    u_mag_exact = u_max * factor
    excepted_profile = normal[:, None] * u_mag_exact[None, :]

    # Create profile and evaluate
    profile = ParaboloidInletProfile(Q, radius, center, normal)
    evaluated_profile = profile(x)

    assert np.linalg.norm(evaluated_profile - excepted_profile) < 1e-10
    
    # Update flow rate and re-test
    u_max = 20
    Q = u_max * np.pi * radius**2 / 2
    profile.update_flow_rate(Q)
    evaluated_profile = profile(x)
    u_mag_exact = u_max * factor
    excepted_profile = normal[:, None] * u_mag_exact[None, :]

    assert np.linalg.norm(evaluated_profile - excepted_profile) < 1e-10

def plot_parabolic_inlet_3d():
    # Define inlet profile
    flow_rate = 2.0
    radius = 1.0
    center = [2, 1, 0]
    normal = [0, 0, 1]

    profile = ParaboloidInletProfile(flow_rate, radius, center, normal)

    # Create a grid in the x-y plane
    num_points = 100
    x = np.linspace(center[0] - radius, center[0] + radius, num_points)
    y = np.linspace(center[1] - radius, center[1] + radius, num_points)
    X, Y = np.meshgrid(x, y)

    # Flatten and stack to (3, num_points^2)
    points = np.vstack([X.ravel(), Y.ravel(), np.full(X.size, center[2])])

    # Evaluate velocity magnitude
    velocities = profile(points)
    V = np.linalg.norm(velocities, axis=0).reshape(X.shape)

    # 3D plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, V, cmap='viridis')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('Velocity [m/s]')
    ax.set_title('Parabolic Inlet Velocity Profile')
    plt.savefig('parabolic_inlet_profile.png', dpi=300)

def plot_parabolic_inlet_2d():
    # Define inlet profile
    flow_rate = 2.0
    radius = 1.0
    center = [0, 0]
    normal = [1, 0]

    profile = ParabolicInletProfile(flow_rate, radius, center, normal)

    # Create a line in the y direction through the center
    num_points = 100
    y = np.linspace(center[1] - radius, center[1] + radius, num_points)
    x = np.full_like(y, center[0])
    points = np.vstack([x, y])

    # Evaluate velocity magnitude
    velocities = profile(points)
    V = np.linalg.norm(velocities, axis=0)

    # 2D plot
    plt.figure(figsize=(8,6))
    plt.plot(y, V, label='Velocity Profile')
    plt.xlabel('y [m]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Parabolic Inlet Velocity Profile')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(center[1] - radius, color='red', linestyle='--', label='Inlet Edge')
    plt.axvline(center[1] + radius, color='red', linestyle='--')
    plt.legend()
    plt.grid()
    plt.savefig('parabolic_inlet_profile_2d.png', dpi=300)

def plot_plug_inlet_2d():
    # Define inlet profile
    speed = 5.0
    center = [1, 1]
    normal = [1, 0]

    profile = PlugInletProfile(speed, normal, center=center, radius=0.5)

    # Create a line in the y direction through the center
    num_points = 100
    y = np.linspace(center[1] - 1.5, center[1] + 1.5, num_points)
    x = np.full_like(y, center[0])
    points = np.vstack([x, y])

    # Evaluate velocity magnitude
    velocities = profile(points)
    V = np.linalg.norm(velocities, axis=0)

    # 2D plot
    plt.figure(figsize=(8,6))
    plt.plot(y, V, label='Velocity Profile')
    plt.xlabel('y [m]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Plug Inlet Velocity Profile')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.legend()
    plt.grid()
    plt.savefig('plug_inlet_profile_2d.png', dpi=300)

def plot_plug_inlet_3d():
    # Define inlet profile
    speed = 5.0
    center = [1, 1, 0]
    normal = [0, 0, 1]

    profile = PlugInletProfile(speed, normal, center=center, radius=0.5)

    # Create a grid in the x-y plane
    num_points = 100
    x = np.linspace(center[0] - 1.5, center[0] + 1.5, num_points)
    y = np.linspace(center[1] - 1.5, center[1] + 1.5, num_points)
    X, Y = np.meshgrid(x, y)

    # Flatten and stack to (3, num_points^2)
    points = np.vstack([X.ravel(), Y.ravel(), np.full(X.size, center[2])])

    # Evaluate velocity magnitude
    velocities = profile(points)
    V = np.linalg.norm(velocities, axis=0).reshape(X.shape)

    # 3D plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, V, cmap='viridis')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('Velocity [m/s]')
    ax.set_title('Plug Inlet Velocity Profile')
    plt.savefig('plug_inlet_profile_3d.png', dpi=300)

if __name__ == '__main__':
    plot_parabolic_inlet_3d()
    plot_parabolic_inlet_2d()
    plot_plug_inlet_2d()
    plot_plug_inlet_3d()
