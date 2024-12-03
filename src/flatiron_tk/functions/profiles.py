from ..info.messages import import_fenics
fe = import_fenics()
import numpy as np

# -----------------------------------------------------

def plug(x, flow_direction, centerline_speed):
    '''
    Constant value in direction flow_direction.
    Here I have `x` as an input to make it consistent with the other profile functions
    '''
    return np.array(flow_direction)*centerline_speed


def parabolic_2d(x, flow_direction, centerline_speed, center, face_radius):

    '''
    Return parabolic profile defined on a line boundary (2D mesh domain). This function
            |
            |
            |
            | flow_direction
    center  |------------> -
            |              |
            |              | face_radius
            |              |
            |              -
    '''

    nn = np.array(flow_direction)
    nn = nn/np.linalg.norm(nn)
    nx = nn[0]
    ny = nn[1]
    tx = ny ; ty = -nx
    xc = np.array(center) ; R = face_radius
    U = centerline_speed
    eta = np.abs(np.dot(x-xc, [tx,ty]))/R

    # Constrain velocity to 0 if it exceed the (normalized) radius
    if eta >= 1:
        u = 0.
    else:
        u = U * (1 + eta) * (1 - eta)
    ux = nx*u ; uy = ny*u
    return np.array([ux, uy])


def parabolic_3d(x, flow_direction, centerline_speed, center, face_radius):

    '''
    Return a parabolic profile defined on a circular boundary (for 3D mesh domain).
    For flow direction `n`, we define the parabolic profile as having
    velocity equal to centerline_speed at the center, and decreases to zero
    at the distance `face_radius` in the direction orthogonal to the flow direction.
    '''

    xc = center
    R = face_radius
    U = centerline_speed
    r = np.linalg.norm(np.array(x) - np.array(xc)) 
    u = U*(R - r)**2/R**2
    return u*np.array(flow_direction)






