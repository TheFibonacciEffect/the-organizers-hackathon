import firedrake as fd
import firedrake_adjoint as fda
import numpy as np
from pyadjoint import stop_annotating, no_annotations
from scipy.spatial import KDTree
# import matplotlib.pyplot as plt
from helpers import (
    alpha,
    calculate_diffusion_coeff_channel_design,
    calculate_heat_transfer_coeff_channel_design,
    calculate_heat_transfer_coeff_substrate,
    calculate_heat_transfer_effective,
    mesh_size,
    ramp_interpolation,
    wall_drag_term,
)
from pyMMAopt import ReducedInequality, MMASolver

# --- 0. Parameters Initialisation ---
# 0.1 Mesh and scales
Nx = 100
Ny = 100
Lx = 1.0 # non-dim
Ly = 1.0 # non-dim

applied_pressure = 10000 # Pa
L = 0.001 # m
U = 20.0 # m/s TODO: compute from pressure drop

mesh = fd.RectangleMesh(Nx, Ny, Lx, Ly)

# 0.2 Physical Data
channel_thickness = 380e-6
Ht = 0.5 * channel_thickness
nondim_Ht = Ht / L

substrate_thickness = 150e-6
Hs = 0.5 * substrate_thickness
nondim_Hs = Hs / L

conductivity_fluid = 0.598
conductivity_substrate = 149
viscosity_fluid = 1.004e-3
capacity_fluid = 4180
density_fluid = 998

Re = L * U * density_fluid / viscosity_fluid
prandtl_number = fd.Constant(capacity_fluid * viscosity_fluid / conductivity_fluid)

# 0.3 Optimisation Parameters
STAGES = 20
iter_per_stage = 20
total_iterations = STAGES * iter_per_stage

ramp_p_fluid_values = np.linspace(300.0, 10.0, STAGES)
ramp_p_thermal_values = np.linspace(40.0, 310.0, STAGES)

p_value = 8

Htfactor_value = 2e2
initial_rho_value = 0.1
cost_function_scale = 100.0

parameters_mma = {
    "move": 0.2,
    "maximum_iterations": iter_per_stage,
    "m": 1,
    "IP": 0,
    "tol": 1e-6,
    "accepted_tol": 1e-4,
    "norm": "L2",
    "gcmma": False,
}

# --- 1. Mesh ---
WALLS = (3, 4)
INLET = 1
OUTLET = 2

x, y = fd.SpatialCoordinate(mesh)
R = fd.FunctionSpace(mesh, "R", 0)

ramp_p_thermal = fd.Function(R)
ramp_p_fluid = fd.Function(R)
Htfactor = fd.Function(R)

with stop_annotating():
    ramp_p_thermal.interpolate(fd.Constant(ramp_p_thermal_values[0]))
    ramp_p_fluid.interpolate(fd.Constant(ramp_p_fluid_values[0]))
    Htfactor.interpolate(fd.Constant(Htfactor_value))

# --- 2.Power Map ---
# The power map is a grid of any resolution, and will be projected on the rectangular mesh
# with a cell size of (Lx/Nx, Ly/Ny)

# TODO: generate the powermap in a random way, maybe use gaussian distribution for less sharp interfaces
# random square moving around (one or two squares)
power_map_data = np.zeros((Nx, Ny))
for i in range(int(Nx / 3), int(4 * Nx / 5)):
    power_map_data[i, int(2 * Ny / 5)] = 1.0
    power_map_data[i, int(4 * Ny / 5)] = 1.0

for i in range(int(Nx / 5), int(3 * Nx / 5)):
    power_map_data[i, int(Ny / 2)] = 1.0

POWER_MAP = fd.FunctionSpace(mesh, "DG", 0)
power_map_f = fd.Function(POWER_MAP)
power_map_grid = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))

XY = fd.VectorFunctionSpace(mesh, POWER_MAP.ufl_element())
with stop_annotating():
    xy = fd.interpolate(mesh.coordinates, XY)

kdtree = KDTree(np.column_stack((power_map_grid[1].ravel(), power_map_grid[0].ravel())))
power_map_f.dat.data[:] = power_map_data.ravel()[kdtree.query(xy.dat.data_ro)[1]]
# TODO: save powermap function term as a field (h5 + pvd)

# fd.tripcolor(power_map_f)
# plt.savefig("power_map.png")

# fd.triplot(mesh)
# plt.legend()
# plt.savefig("mesh.png")

# --- 3. Filter Problem ---
RHO = fd.FunctionSpace(mesh, "DG", 0)
RHOF = fd.FunctionSpace(mesh, "CG", 1)

rho = fd.Function(RHO, name="Volume Fraction")
with stop_annotating():
    rho.interpolate(fd.Constant(initial_rho_value))

rhof = fd.Function(RHOF, name="Filtered Volume Fraction")
rhof_test = fd.TestFunction(RHOF)

hmax = mesh_size(mesh, mode="max")
filter_radius = 0.5 * hmax**2

F_filter = (
    filter_radius * fd.inner(fd.grad(rhof), fd.grad(rhof_test)) * fd.dx(domain=mesh)
    + rhof * rhof_test * fd.dx(domain=mesh)
    - rho * rhof_test * fd.dx(domain=mesh)
)

fd.solve(F_filter == 0, rhof)

# --- 4. Fluid Problem ---

# Define Taylor--Hood of order 2 function space UP
V = fd.VectorFunctionSpace(mesh, "CG", 2)
Q = fd.FunctionSpace(mesh, "CG", 1)
UP = fd.MixedFunctionSpace([V, Q])
print("velocity-pressure dofs: ", UP.dim())

# Define Function and TestFunction(s)
up = fd.Function(UP, name="Solution")
(u, p) = fd.split(up)
vq = fd.TestFunction(UP)
(v, q) = fd.split(vq)

# Define viscosity and bcs
nu = fd.Constant(1.0 / Re)

p0 = fd.Constant(1.0)  # 1 at left, 0 at right
bcs = fd.DirichletBC(UP.sub(0), fd.Constant((0, 0)), WALLS)
n = fd.FacetNormal(mesh)

# Stokes problem
F_fluid = (
    fd.inner(2 * nu * fd.sym(fd.grad(u)), fd.sym(fd.grad(v))) * fd.dx
    - fd.div(u) * q * fd.dx
    - fd.div(v) * p * fd.dx
    + p0 * fd.dot(v, n) * fd.ds(INLET)
)

# Brinkman term
Ht = 0.5 * channel_thickness

# TODO: save brinkman term as a field
F_fluid += (
    nu
    * (
        wall_drag_term(Ht=Ht, L=L)
        + alpha(rhof, Ht=Ht, ramp_p=ramp_p_fluid, Htfactor=Htfactor, L=L)
    )
    * fd.inner(u, v)
    * fd.dx
)

# Solve Stokes  problem
fd.solve(F_fluid == 0, up, bcs)

# fd.tripcolor(up.sub(0))
# plt.savefig("velocity.png")

# --- 5. Thermal Problem ---
# 5.1 Thermal Coefficients
# 5.1.1 For the Channel Layer
advection_coeff = (24535.0 / 33462.0) * ramp_interpolation(rhof, ramp_p_fluid)

conductivity_channel_coeff = calculate_diffusion_coeff_channel_design(
    conductivity_fluid, conductivity_substrate, rhof, ramp_p_fluid
)

diffusion_channel_coeff = (
    conductivity_channel_coeff / conductivity_fluid * fd.Constant(nu / prandtl_number)
)

heat_transfer_coeff_channel_design = calculate_heat_transfer_coeff_channel_design(
    conductivity_solid=conductivity_substrate,
    conductivity_fluid=conductivity_fluid,
    channel_thickness=channel_thickness,
    rhof=rhof,
    ramp_p=ramp_p_thermal,
)
heat_transfer_coeff_substrate = calculate_heat_transfer_coeff_substrate(
    conductivity_substrate, 0.5 * substrate_thickness
)
heat_transfer_coeff = calculate_heat_transfer_effective(
    heat_transfer_coeff_channel_design, heat_transfer_coeff_substrate
)

coupling_channel_to_bottom_layer = (
    heat_transfer_coeff
    * 0.5
    * channel_thickness
    / fd.Constant(2.0 * nondim_Ht**2 * conductivity_fluid)
    * fd.Constant(nu / prandtl_number)
)


# 5.1.2 For the die Substrate Layer
diffusion_die_substrate_coeff = fd.Constant(4.0 / 3.0)
coupling_die_substrate_to_top_layer = (
    heat_transfer_coeff
    * 0.5
    * substrate_thickness
    / fd.Constant(2.0 * nondim_Hs**2 * conductivity_substrate)
)


Z = fd.FunctionSpace(mesh, "CG", 1)
T = Z * Z
t_total = fd.Function(T, name="Temperature")
tau_total = fd.TestFunction(T)
channel_temperature, substrate_temperature = fd.split(t_total)
tau, psi = fd.split(tau_total)

# 5.2 Thermal Form
# 5.2.1 Heat Flux
F_thermal = -power_map_f * psi * fd.dx
# 5.2.2 Die Substrate Layer
# 5.2.2.1 Die Substrate layer diffusion
F_thermal += (
    diffusion_die_substrate_coeff
    * fd.inner(fd.grad(substrate_temperature), fd.grad(psi))
    * fd.dx
)

# 5.2.2.2 Die Substrate layer coupling to Channel layer
F_thermal += (
    coupling_die_substrate_to_top_layer
    * (substrate_temperature - channel_temperature)
    * psi
    * fd.dx
)

# 5.2.3 Channel Layer
# 5.2.3.1 Channel Layer advection
F_thermal += advection_coeff * fd.inner(u, fd.grad(channel_temperature)) * tau * fd.dx

# 5.2.3.2 Channel Layer diffusion
F_thermal += (
    diffusion_channel_coeff
    * fd.inner(fd.grad(channel_temperature), fd.grad(tau))
    * fd.dx
)

# 5.2.3.3 Channel Layer coupling to Substrate layer
F_thermal += (
    -coupling_channel_to_bottom_layer
    * (substrate_temperature - channel_temperature)
    * tau
    * fd.dx
)

# 5.3 Thermal Boundary Conditions
bc1 = fd.DirichletBC(T.sub(0), fd.Constant(0.0), INLET)
fd.solve(F_thermal == 0, t_total, bcs=[bc1])

# --- 6. MMA ---

# 6.1 Vizualisation callback
output_file = fd.File("output/output.pvd")

t_total_node = fda.Control(t_total)
t_total_viz = fd.Function(T)
t_total_viz.sub(0).rename("channel_temperature")
t_total_viz.sub(1).rename("substrate_temperature")

up_node = fda.Control(up)
up_viz = fd.Function(UP)
up_viz.sub(0).rename("velocity")
up_viz.sub(1).rename("pressure")

rho_node = fda.Control(rho)
rho_viz = fd.Function(RHO)
rho_viz.rename("rho")

rhof_node = fda.Control(rhof)
rhof_viz = fd.Function(RHOF)
rhof_viz.rename("rhof")

global_i = 0
@no_annotations
def output(x, y, z):
    global global_i
    # TODO: save output to hdf5 file 
    if global_i % 10 == 0:
        _u, _p = up_node.tape_value().split()
        up_viz.sub(0).assign(_u)
        up_viz.sub(1).assign(_p)

        _channel_temperature, _substrate_temperature = t_total_node.tape_value().split()
        t_total_viz.sub(0).assign(_channel_temperature)
        t_total_viz.sub(1).assign(_substrate_temperature)

        rho_viz.assign(rho_node.tape_value())
        rhof_viz.assign(rhof_node.tape_value())

        output_file.write(
            up_viz.sub(0),
            up_viz.sub(1),
            t_total_viz.sub(0),
            t_total_viz.sub(1),
            rho_viz,
            rhof_viz,
        )


    global_i += 1
    return y


# 6.2 Cost function definition
with stop_annotating():
    p = fd.Function(R).interpolate(fd.Constant(p_value))

J = cost_function_scale * fd.assemble(substrate_temperature**p * fd.dx) ** (
    1.0 / p_value
)

c = fda.Control(rho)
Jhat = fda.ReducedFunctional(
    J,
    c,
    derivative_cb_post=output,
)

# 6.3 Volume Constraint definition
K = fd.assemble(rhof * fd.dx)
Klimit = fd.assemble(fd.Constant(1.0) * fd.dx(mesh)) * 2.0
Kcontrol = fda.Control(K)
Khat = fda.ReducedFunctional(K, c)

# 6.4 Continuation loop
for i in range(STAGES):
    ramp_p_thermal.interpolate(fd.Constant(ramp_p_thermal_values[i]))
    ramp_p_fluid.interpolate(fd.Constant(ramp_p_fluid_values[i]))

    problem = fda.MinimizationProblem(
        Jhat,
        bounds=(0.0, 1.0),
        constraints=[
            ReducedInequality(Khat, Klimit, Kcontrol, normalized=False),
        ],
    )

    solver = MMASolver(problem, parameters=parameters_mma)
    results = solver.solve()

    rho_opt = results["control"]
    with stop_annotating():
        rho.assign(rho_opt)