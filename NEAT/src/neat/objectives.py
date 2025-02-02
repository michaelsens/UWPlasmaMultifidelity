""" Objectives module of NEAT

This script defined the necessary classes and
functions to be used in an optimization loop
driven by the SIMSOPT package. One of the main
goals is the definition of the objective functions
that are minimized during the optimization loop. This
script makes heavy use of SIMSOPT's Optimizable class.

"""

from typing import Union

import numpy as np
from qic import Qic
from qsc import Qsc

try:
    from simsopt._core.optimizable import Optimizable
    from simsopt.objectives import LeastSquaresProblem
    from simsopt.solve import least_squares_mpi_solve, least_squares_serial_solve
    from simsopt.util import MpiPartition

    simsopt_available = True
except ImportError as error:
    print("simsopt not avaiable")
    simsopt_available = False

from neat.tracing import ParticleEnsembleOrbit

base_class = (
    Optimizable if simsopt_available else object
)  # Dynamically select the base class


class LossFractionResidual(base_class):
    """
    Objective function for optimization.
    The residual here is the loss fraction of
    particles traced for a given amount of time.
    """

    def __init__(
        self,
        field: Union[Qsc, Qic],
        particles: ParticleEnsembleOrbit,
        nsamples=500,
        tfinal=0.0003,
        nthreads=2,
        r_max=0.12,
    ) -> None:
        self.field = field
        self.particles = particles
        self.nsamples = nsamples
        self.tfinal = tfinal
        self.nthreads = nthreads
        self.r_max = r_max

        if simsopt_available:
            super().__init__(depends_on=[self.field])

    def compute(self):
        """Calculate the loss fraction"""
        self.orbits = ParticleEnsembleOrbit(  # pylint: disable=W0201
            self.particles, self.field, self.nsamples, self.tfinal, self.nthreads
        )
        self.orbits.loss_fraction(r_max=self.r_max)

    def J(self):
        """Calculate the objective function residual"""
        self.compute()
        return self.orbits.loss_fraction_array[-1]

class MultifidelityLossFractionResidual(LossFractionResidual):
    def __init__(self, field, particles, nsamples_low, nsamples_high, tfinal, nthreads, r_max, switch_frequency):
        super().__init__(field, particles, nsamples_low, tfinal, nthreads, r_max)
        self.nsamples_low = nsamples_low
        self.nsamples_high = nsamples_high
        self.switch_frequency = switch_frequency
        self.iteration = 0
        self.bias_term = 0.0

    def compute(self):
        if self.iteration % self.switch_frequency == 0:
            self.nsamples = self.nsamples_high
            super().compute()
            high_fidelity_residual = self.orbits.loss_fraction_array[-1]

            self.nsamples = self.nsamples_low
            super().compute()
            low_fidelity_residual = self.orbits.loss_fraction_array[-1]

            self.bias_term = high_fidelity_residual - low_fidelity_residual
            self.current_residual = high_fidelity_residual
        else:
            self.nsamples = self.nsamples_low
            super().compute()
            low_fidelity_residual = self.orbits.loss_fraction_array[-1]

            self.current_residual = low_fidelity_residual + self.bias_term

    def J(self):
        self.compute()
        self.iteration += 1
        return self.current_residual



class EffectiveVelocityResidual(base_class):
    """
    Objective function for optimization.
    The residual here is the effective velocity of
    particles traced for a given amount of time.
    This residual is smoother than the loss fraction
    residual and is defined as delta s / delta t.

    - delta s = maximum radial distance travelled by each particle before
        coliding with the wall or reaching the end of the simulation
    - delta t = time until particle collided or until the end of simulation, depends on the particle
    - J = delta s/ delta t or delta s^2/delta t

    """

    def __init__(
        self,
        field: Union[Qsc, Qic],
        particles: ParticleEnsembleOrbit,
        nsamples=500,
        tfinal=0.0003,
        nthreads=2,
        r_max=0.12,
        constant_b20=True,
    ) -> None:
        self.field = field
        self.particles = particles
        self.nsamples = nsamples
        self.tfinal = tfinal
        self.nthreads = nthreads
        self.r_max = r_max
        self.constant_b20 = constant_b20

        if simsopt_available:
            super().__init__(depends_on=[self.field])

    def compute(self):
        """Calculate the effective velocity"""
        self.orbits = ParticleEnsembleOrbit(  # pylint: disable=W0201
            self.particles,
            self.field,
            self.nsamples,
            self.tfinal,
            self.nthreads,
            constant_b20=self.constant_b20,
        )
        self.orbits.loss_fraction(r_max=self.r_max)

        def radial_pos_of_particles(i, particle_pos):
            if self.orbits.lost_times_of_particles[i] == 0:
                return max(particle_pos)  # subtract min(particle_pos), particle_pos[0]?
            return (
                self.r_max
            )  # a bit more accurate -> particle_pos[np.argmax(particle_pos > self.r_max)]

        maximum_radial_pos_of_particles = np.array(
            [
                radial_pos_of_particles(i, particle_pos)
                for i, particle_pos in enumerate(self.orbits.r_pos)
            ]
        )

        tfinal = max(self.orbits.time)
        time_of_particles = np.array(
            [
                (
                    tfinal
                    if self.orbits.lost_times_of_particles[i] == 0
                    else self.orbits.lost_times_of_particles[i]
                )
                for i in range(self.orbits.nparticles)
            ]
        )

        self.effective_velocity = (  # pylint: disable=W0201
            maximum_radial_pos_of_particles / time_of_particles
        )

    def J(self):
        """Calculate the objective function residual"""
        self.compute()
        return 1e-5 * self.effective_velocity / np.sqrt(self.orbits.nparticles)


class OptimizeLossFractionSkeleton(base_class):
    """
    Skeleton of a class used to optimize a given
    objective function using SIMSOPT.
    """

    def __init__(
        self,
        field,
        particles,
        r_max=0.12,
        nsamples=800,
        tfinal=0.0001,
        nthreads=2,
    ) -> None:
        # log(level=logging.DEBUG)

        self.field = field
        self.particles = particles
        self.nsamples = nsamples
        self.tfinal = tfinal
        self.nthreads = nthreads
        self.r_max = r_max

        self.residual = LossFractionResidual(
            self.field,
            self.particles,
            self.nsamples,
            self.tfinal,
            self.nthreads,
            self.r_max,
        )

        self.field.fix_all()
        # self.field.unfix("etabar")
        # self.field.unfix("rc(1)")
        # self.field.unfix("zs(1)")
        self.field.unfix("rc(2)")
        # self.field.unfix("zs(2)")
        self.field.unfix("rc(3)")
        # self.field.unfix("zs(3)")
        # self.field.unfix("B2c")

        # Define objective function
        self.prob = LeastSquaresProblem.from_tuples(
            [
                (self.residual.J, 0, 1),
                # (self.field.get_elongation, 0.0, 3),
                # (self.field.get_inv_L_grad_B, 0, 2),
                # (self.field.get_grad_grad_B_inverse_scale_length_vs_varphi, 0, 2),
                # (self.field.get_B20_mean, 0, 0.01),
            ]
        )

    def run(self, ftol=1e-6, n_iterations=100):
        """Run the optimization problem defined in this class in serial"""
        print("Starting optimization in serial")
        if simsopt_available:
            least_squares_serial_solve(self.prob, ftol=ftol, max_nfev=n_iterations)
        else:
            print("Currently optimization with run() only available with simsopt")

    def run_parallel(self, n_iterations=100, rel_step=1e-3, abs_step=1e-5):
        """Run the optimization problem defined in this class in parallel"""
        if simsopt_available:
            self.mpi = MpiPartition()  # pylint: disable=W0201
            if self.mpi.proc0_world:
                print("Starting optimization in parallel")
            least_squares_mpi_solve(
                self.prob,
                self.mpi,
                grad=True,
                rel_step=rel_step,
                abs_step=abs_step,
                max_nfev=n_iterations,
            )
        else:
            print("Currently optimization with run() only available with simsopt")

class OptimizeMultifidelityLossFractionSkeleton(base_class):
    """
    Skeleton of a class used to optimize a given
    objective function using SIMSOPT with multifidelity models.
    """

    def __init__(
        self,
        field,
        particles,
        r_max=0.12,
        nsamples_low=100,
        nsamples_high=1000,
        tfinal=0.0001,
        nthreads=2,
        switch_frequency=10,
    ) -> None:
        self.field = field
        self.particles = particles
        self.nsamples_low = nsamples_low
        self.nsamples_high = nsamples_high
        self.tfinal = tfinal
        self.nthreads = nthreads
        self.r_max = r_max
        self.switch_frequency = switch_frequency

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
        self.field.unfix("rc(2)")
        self.field.unfix("rc(3)")

        # Define objective function
        self.prob = LeastSquaresProblem.from_tuples(
            [
                (self.residual.J, 0, 1),
            ]
        )

    def run(self, ftol=1e-6, n_iterations=100):
        """Run the optimization problem defined in this class in serial"""
        print("Starting optimization in serial")
        if simsopt_available:
            least_squares_serial_solve(self.prob, ftol=ftol, max_nfev=n_iterations)
        else:
            print("Currently optimization with run() only available with simsopt")

    def run_parallel(self, n_iterations=100, rel_step=1e-3, abs_step=1e-5):
        """Run the optimization problem defined in this class in parallel"""
        if simsopt_available:
            self.mpi = MpiPartition()  # pylint: disable=W0201
            if self.mpi.proc0_world:
                print("Starting optimization in parallel")
            least_squares_mpi_solve(
                self.prob,
                self.mpi,
                grad=True,
                rel_step=rel_step,
                abs_step=abs_step,
                max_nfev=n_iterations,
            )
        else:
            print("Currently optimization with run() only available with simsopt")
