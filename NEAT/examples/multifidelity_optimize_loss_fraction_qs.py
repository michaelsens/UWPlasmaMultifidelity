#!/usr/bin/env python3

import glob
import os
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve, least_squares_serial_solve
from simsopt.util import MpiPartition

from neat.fields import StellnaQS
from neat.objectives import EffectiveVelocityResidual, LossFractionResidual, MultifidelityLossFractionResidual
from neat.tracing import ChargedParticle, ChargedParticleEnsemble, ParticleOrbit

r_initial = 0.05
r_max = 0.1
n_iterations = 20
ftol = 1e-5
B0 = 5
B2c = B0 / 7
nsamples_low = 200
nsamples_high = 600
tfinal = 6e-5
stellarator_index = 2
constant_b20 = True
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
ntheta = 10  # resolution in theta
nphi = 4  # resolution in phi
nlambda_trapped = 14  # number of pitch angles for trapped particles
nlambda_passing = 2  # number of pitch angles for passing particles
nthreads = 4
switch_frequency = 10

class optimize_multifidelity_loss_fraction:
    def __init__(
        self,
        field,
        particles,
        r_max=r_max,
        nsamples_low=nsamples_low,
        nsamples_high=nsamples_high,
        tfinal=tfinal,
        nthreads=nthreads,
        switch_frequency=switch_frequency,
        parallel=False,
    ) -> None:
        self.field = field
        self.particles = particles
        self.nsamples_low = nsamples_low
        self.nsamples_high = nsamples_high
        self.tfinal = tfinal
        self.nthreads = nthreads
        self.r_max = r_max
        self.switch_frequency = switch_frequency
        self.parallel = parallel

        self.mpi = MpiPartition()

        self.residual = MultifidelityLossFractionResidual(
            self.field,
            self.particles,
            self.nsamples_low,
            self.nsamples_high,
            self.tfinal,
            self.nthreads,
            self.r_max,
            self.switch_frequency,
        )

        self.field.fix_all()
        self.field.unfix("etabar")
        self.field.unfix("rc(1)")
        self.field.unfix("zs(1)")
        self.field.unfix("rc(2)")
        self.field.unfix("zs(2)")
        self.field.unfix("rc(3)")
        self.field.unfix("zs(3)")
        self.field.unfix("B2c")

        # Define objective function
        self.prob = LeastSquaresProblem.from_tuples(
            [
                (self.residual.J, 0, 40),
                (self.field.get_grad_grad_B_inverse_scale_length_vs_varphi, 0, 0.01),
                (self.field.get_B20_mean, 0, 0.01),
            ]
        )

    def run(self, ftol=1e-6, n_iterations=100, rel_step=1e-3, abs_step=1e-5):
        # Algorithms that do not use derivatives
        # Relative/Absolute step size ~ 1/n_particles
        # with MPI, to see more info do mpi.write()
        if self.parallel:
            least_squares_mpi_solve(
                self.prob,
                self.mpi,
                grad=True,
                rel_step=rel_step,
                abs_step=abs_step,
                max_nfev=n_iterations,
            )
        else:
            least_squares_serial_solve(self.prob, ftol=ftol, max_nfev=n_iterations)

g_field = StellnaQS.from_paper(stellarator_index, nphi=151, B2c=B2c, B0=B0)
g_particle = ChargedParticleEnsemble(
    r_initial=r_initial,
    r_max=r_max,
    energy=energy,
    charge=charge,
    mass=mass,
    ntheta=ntheta,
    nphi=nphi,
    nlambda_trapped=nlambda_trapped,
    nlambda_passing=nlambda_passing,
)
optimizer = optimize_multifidelity_loss_fraction(
    g_field,
    g_particle,
    r_max=r_max,
    tfinal=tfinal,
    nsamples_low=nsamples_low,
    nsamples_high=nsamples_high,
    switch_frequency=switch_frequency,
)
test_particle = ChargedParticle(
    r_initial=r_initial, theta_initial=np.pi / 2, phi_initial=np.pi, Lambda=0.98
)
##################
if optimizer.mpi.proc0_world:
    print("Before run:")
    print(" Iota: ", optimizer.field.iota)
    print(" Max elongation: ", max(optimizer.field.elongation))
    print(" Max Inverse L grad B: ", max(optimizer.field.inv_L_grad_B))
    print(
        " Max Inverse L gradgrad B: ", optimizer.field.grad_grad_B_inverse_scale_length
    )
    print(" Initial Mean residual: ", np.mean(optimizer.residual.J()))
    print(" Initial Loss Fraction: ", optimizer.residual.orbits.loss_fraction_array[-1])
    print(" Objective function: ", optimizer.prob.objective())
    print(" Initial equilibrium: ")
    print(
        "        rc      = [", ",".join([str(elem) for elem in optimizer.field.rc]), "]"
    )
    print(
        "        zs      = [", ",".join([str(elem) for elem in optimizer.field.zs]), "]"
    )
    print("        etabar = ", optimizer.field.etabar)
    print("        B2c = ", optimizer.field.B2c)
    print("        B20 = ", optimizer.field.B20_mean)
    optimizer.residual.orbits.plot_loss_fraction(show=False)
initial_orbit = ParticleOrbit(test_particle, g_field, nsamples=nsamples_high, tfinal=tfinal)
initial_field = StellnaQS.from_paper(stellarator_index, nphi=151, B2c=B2c, B0=B0)
##################
start_time = time.time()

optimizer.run(ftol=ftol, n_iterations=n_iterations)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nOptimization took {elapsed_time:.2f} seconds\n")
##################
if optimizer.mpi.proc0_world:
    print("After run:")
    print(" Iota: ", optimizer.field.iota)
    print(" Max elongation: ", max(optimizer.field.elongation))
    print(" Max Inverse L grad B: ", max(optimizer.field.inv_L_grad_B))
    print(
        " Max Inverse L gradgrad B: ", optimizer.field.grad_grad_B_inverse_scale_length
    )
    print(" Final Mean residual: ", np.mean(optimizer.residual.J()))
    print(" Final Loss Fraction: ", optimizer.residual.orbits.loss_fraction_array[-1])
    print(" Objective function: ", optimizer.prob.objective())
    print(" Final equilibrium: ")
    print(
        "        rc      = [", ",".join([str(elem) for elem in optimizer.field.rc]), "]"
    )
    print(
        "        zs      = [", ",".join([str(elem) for elem in optimizer.field.zs]), "]"
    )
    print("        etabar = ", optimizer.field.etabar)
    print("        B2c = ", optimizer.field.B2c)
    print("        B20 = ", optimizer.field.B20_mean)
    optimizer.residual.orbits.plot_loss_fraction(show=False)
    initial_patch = mpatches.Patch(color="#1f77b4", label="Initial")
    final_patch = mpatches.Patch(color="#ff7f0e", label="Final")
    plt.legend(handles=[initial_patch, final_patch])
final_orbit = ParticleOrbit(test_particle, g_field, nsamples=nsamples_high, tfinal=tfinal)
final_field = g_field
##################
plt.figure()
plt.plot(
    initial_orbit.r_pos * np.cos(initial_orbit.theta_pos),
    initial_orbit.r_pos * np.sin(initial_orbit.theta_pos),
    label="Initial Orbit",
)
plt.plot(
    final_orbit.r_pos * np.cos(final_orbit.theta_pos),
    final_orbit.r_pos * np.sin(final_orbit.theta_pos),
    label="Final Orbit",
)
plt.gca().set_aspect("equal", adjustable="box")
plt.legend()
plt.xlabel(r"r cos($\theta$)")
plt.ylabel(r"r sin($\theta$)")
plt.tight_layout()
initial_orbit.plot_orbit_3d(show=False, r_surface=r_max)
final_orbit.plot_orbit_3d(show=False, r_surface=r_max)
# initial_orbit.plot_animation(show=False)
# final_orbit.plot_animation(show=True)
plt.show()

# Remove output files from simsopt
for f in glob.glob("residuals_202*"):
    os.remove(f)
for f in glob.glob("simsopt_202*"):
    os.remove(f)
