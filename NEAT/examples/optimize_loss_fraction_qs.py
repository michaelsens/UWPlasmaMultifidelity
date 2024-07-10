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
from neat.objectives import EffectiveVelocityResidual, LossFractionResidual
from neat.tracing import ChargedParticle, ChargedParticleEnsemble, ParticleOrbit


initialArr = np.linspace(-2, 2, 25)


B2c = -25
rc0 = 0.777
rc1 = 1.666
rc2 = -0.1

r_initial = 0.05
r_max = 0.1
n_iterations = 50
ftol = 1e-7
B0 = 5
nsamples = 1000
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


class optimize_loss_fraction:
    def __init__(
        self,
        field,
        particles,
        r_max=r_max,
        nsamples=nsamples,
        tfinal=tfinal,
        nthreads=nthreads,
        parallel=False,
        constant_b20=constant_b20,
    ) -> None:
        self.field = field
        self.particles = particles
        self.nsamples = nsamples
        self.tfinal = tfinal
        self.nthreads = nthreads
        self.r_max = r_max
        self.parallel = parallel

        results = []

        self.mpi = MpiPartition()
        print("Initial: " + str(self.field.zs[1]))
        for val in initialArr:
            self.field.zs[1] = val
            self.field.calculate()
            
            

            # self.residual = LossFractionResidual(
            self.residual = EffectiveVelocityResidual(
                # LossFractionResidual(
                self.field,
                self.particles,
                self.nsamples,
                self.tfinal,
                self.nthreads,
                self.r_max,
                constant_b20=constant_b20,
            )

            self.field.fix_all()
            #self.field.unfix("etabar")
            #self.field.unfix("rc(0)")
            #self.field.unfix("zs(0)")
            #self.field.unfix("rc(1)")
            self.field.unfix("zs(1)")
            #self.field.unfix("rc(2)")
            #self.field.unfix("zs(2)")
            
            ####
    
            
            #self.field.unfix("B2c")

            # Define objective function
            self.prob = LeastSquaresProblem.from_tuples(
                [
                    (self.residual.J, 0, 1),
                    # (self.field.get_elongation, 0.0, 0.5),
                    # (self.field.get_inv_L_grad_B, 0, 0.1),
                    (self.field.get_grad_grad_B_inverse_scale_length_vs_varphi, 0, 0.01),
                    (self.field.get_B20_mean, 0, 0.01),
                ]
            )

            result = np.sum((self.prob.residuals())**2)
            print(str(result) + "\t" + str(val))
            results.append((val, result))

        values, result_values = zip(*results)

        plt.figure()
        plt.plot(values, result_values, marker='o')
        plt.xlabel('zs0')
        plt.ylabel('Sum of Squared Residuals')
        plt.title('Effect of zs0 on Sum of Squared Residuals')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        exit()

        


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


g_field = StellnaQS(rc=[rc0, rc1, rc2], zs=[0, 0.154, 0.0111], nfp=2, etabar=0.64, order='r2', B2c=B2c, B0=B0)
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
optimizer = optimize_loss_fraction(
    g_field,
    g_particle,
    r_max=r_max,
    tfinal=tfinal,
    nsamples=nsamples,
    constant_b20=constant_b20,
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
initial_orbit = ParticleOrbit(test_particle, g_field, nsamples=nsamples, tfinal=tfinal)
initial_field = StellnaQS(rc=[rc1, rc2, 0.0102], zs=[0, 0.154, 0.0111], nfp=2, etabar=0.64, order='r2', B2c=B2c, B0=B0)

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
final_orbit = ParticleOrbit(test_particle, g_field, nsamples=nsamples, tfinal=tfinal)
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
