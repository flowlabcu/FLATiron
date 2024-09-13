from dolfin import *

if MPI.size(MPI.comm_world) > 1:
    info("Sorry, this demo does not (yet) run in parallel.")
    exit(0)

# Generate meshes
background_mesh = UnitSquareMesh(16, 16)
annulus_mesh = Mesh("../donut.xml.gz")

center = Point(0.5, 0.5)
r = 0.2

# Build the multimesh
multimesh = MultiMesh()
multimesh.add(background_mesh)
multimesh.add(annulus_mesh)
multimesh.build()

# Identify background cells within the hole and mark them as covered
multimesh.auto_cover(0, center)

# Variational formulation
V = MultiMeshFunctionSpace(multimesh, "P", 1)
u, v = TrialFunction(V), TestFunction(V)
n = FacetNormal(multimesh)
h = 2*Circumradius(multimesh)

a = (inner(grad(u), grad(v)) * dX
        - inner(avg(grad(u)), jump(v, n)) * dI
        - inner(avg(grad(v)), jump(u, n)) * dI
        + Constant(10) / avg(h) * jump(u) * jump(v) * dI
        + inner(jump(grad(u)), jump(grad(v))) * dO)

L = Constant(1) * v * dX

# Assemble system
A = assemble_multimesh(a)
b = assemble_multimesh(L)

# Will mark boundary of outer mesh
class OuterBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
outer = OuterBoundary()
mf0 = MeshFunction("size_t", multimesh.part(0), multimesh.part(0).topology().dim() - 1)
outer.mark(mf0, 2)
bc0 = MultiMeshDirichletBC(V, Constant(0), mf0, 2, 0)

# Will mark inner part of top mesh
class InnerBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((x[0] - 0.5)**2 + (x[1] - 0.5)**2 < 0.5*r)
inner = InnerBoundary()
mf1 = MeshFunction("size_t", multimesh.part(1), multimesh.part(1).topology().dim()-1)
inner.mark(mf1 , 3)
bc1 = MultiMeshDirichletBC(V, Constant(1), mf1, 3, 1)
bcs = [bc0, bc1]

[bc.apply(A, b) for bc in bcs]
V.lock_inactive_dofs(A, b)

# Solve and plot
uh = MultiMeshFunction(V)
x = uh.vector()

solve(A, x, b)

outfile0 = XDMFFile("output/u0.xdmf")
outfile1 = XDMFFile("output/u1.xdmf")

outfile0.write(uh.part(0, deepcopy=True), 0.0)
outfile1.write(uh.part(1, deepcopy=True), 0.0)

outfile0.close()
outfile1.close()

