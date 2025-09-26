import numpy as np 
from flatiron_tk.functions import ParabolicInletProfile
from flatiron_tk.functions import ParaboloidInletProfile

def test_parabolic():
    num_points = 100 

    center = np.array([0.0, 0.5])
    radius = 0.3
    normal = np.array([1.0, 0.0, 0.0])

    u_max = 10 
    Q = u_max * radius * 4 / 3 

    # Sample points along a vertical line through the center
    # Expanding the line to ensure we have points outside the radius (to test zero velocity)
    y_vals = np.linspace(center[1] - (1.2 * radius) , center[1] + (1.2 * radius), num_points)
    x = np.vstack([np.full_like(y_vals, center[0]), y_vals])
    
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
