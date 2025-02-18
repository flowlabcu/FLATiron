'''
Demo for 1D transient convection-diffusion equation on an interval [0,12800]
with no reactions
dc/dt = D*d^2c/dx^2 - u*dc/dx - f

The following problem was taken from "Problem 1" from "Benchmarks for the Transport Equation:
The Convection_Diffusion Forum and Beyond" by Baptista and Adams, 1995

D = 2
f = 0
u = 1.5*sin(2*pi*t/9600)
The Gaussian Source Solution is:
c(x,t) = sigma_0/sigma * exp(-(x-x_bar)^2 / 2*sigma^2)
sigma^2 = sigma_0^2 + 2*D*t
x_bar = x_0 + int(u(T)dT) from 0 to T

This demo demonstrates how to do a transient convection-diffusion problem in flatiron_tk

Author: njrovito
'''

# ------------------------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
import fenics as fe
from flatiron_tk.physics import TransientScalarTransport
from flatiron_tk.mesh import Mesh, LineMesh
from flatiron_tk.solver import PhysicsSolver

class DisconValue(fe.UserExpression):
    def __init__(self, val_outside, val_inside, mid_point, span, eps=1e-16, vector=False, **kwargs):
        self.val_outside = val_outside
        self.val_inside = val_inside
        self.mid_point = mid_point
        self.span = span
        self.eps = eps
        self.vector = vector
        super().__init__(**kwargs)
    def eval(self, value, x):
        xc = self.mid_point
        if x[0] < xc:
            sdf = x[0] - (xc - self.span)
            value[0] = 1/(1 + np.exp(sdf/self.eps))
            dist = self.val_outside - self.val_inside
            value[0] = value[0] * dist + self.val_inside
        else:
            sdf = x[0] - (xc + self.span)
            value[0] = 1/(1 + np.exp(sdf/self.eps))
            dist = self.val_inside - self.val_outside
            value[0] = value[0] * dist + self.val_outside
        return value
    def value_shape(self):
        if self.vector:
            return (2, )
        else:
            return ()

class SinPulse(fe.UserExpression):
    def __init__(self, l, **kwargs):
        self.l = l
        super().__init__(self, **kwargs)
    def eval(self, value, x):
        l = self.l
        if x[0] <= l:
            value[0] = np.sin(np.pi*x[0]/l)
        else:
            value[0] = 0
        return value
    def valuse_shape(self):
        return ( )

# -- Setup mesh and function space
# ne = 0.5e3
# h = np.pi/(int(ne)+1)
# L = np.pi
ne = 256
L = 1
h = L/ne
mesh = LineMesh(0, L, h)

# -- Advection speed beta and diffusivity epsilon
dt = 1e-3
midpoint = 3*L/4
span = L/4
theta = 0.5
beta = DisconValue(1, 1e-4, midpoint, span)
epsilon = DisconValue(1e-4, 1, midpoint, span)

# Define problem
st = TransientScalarTransport(mesh, dt, theta=theta, tag='c')
st.set_element('CG', 1)
st.build_function_space()

# Diffusivity (here set as a constant)
D = st.set_diffusivity(epsilon, epsilon)

# For the velocity term, we have a time-dependent velocity.
# We will create two separate functions u0  and un and update
# them with the appropriate t.
st.set_advection_velocity(beta, beta)

# Similarly, we create f0 and fn for the reaction term (here set to zero)
st.set_reaction(0, 0)

# Set weak form
st.set_weak_form()
st.add_stab()
gamma = h*h/2
st.add_edge_stab(gamma)

# Set bc
bc_dict = {1:{'type': 'dirichlet', 'value': fe.Constant(0.)}}
st.set_bcs(bc_dict)
st.set_initial_condition(fe.interpolate(SinPulse(1/8), st.V))

# Set problem
solver = PhysicsSolver(st)

# Set writer 
st.set_writer('output', 'h5')

# Begin transient section
nt = 6000
t = 0
for i in range(nt):

    # Solve
    solver.solve()

    # Write output
    t += dt
    st.write(time_stamp=t)

    # Update previous solution
    st.update_previous_solution()

    # Update time
    if i%10 == 0:
        c = st.solution_function()
        fe.plot(c)
        plt.ylim([-0.5, 1.5])
        plt.grid(True)
        plt.title(str(i))
        plt.pause(0.01)
        plt.cla()




