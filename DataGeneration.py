import numpy as np
from scipy.interpolate import interp1d
from numba import njit
import time
import os
from joblib import Parallel, delayed
from typing import Union


def generate_function(n_points, f0, f1, max_offsets=(1.3, 0.5, 0.3, 0.05)):
    """
    Generates values of a random function f on the interval [0, 1] with boundary conditions f(0)=f0 and f(1)=f1
    by dividing the interval on smaller subsections and iteratively applying random perturbations to function values
    on these subsections, starting from the linear function f. Smoothness of f is achieved with quadratic interpolation.

    :param n_points: desired length of the output (discretization number)
    :param f0: desired value of f(0)
    :param f1: desired value of f(1)
    :param max_offsets: maximal perturbation offsets for each iteration.
        The length of this touple also determines the number of iterations.
    :return: values of the random function in n_points discrete points on the interval [0, 1]
    """

    n_op = n_points // 2  # we operate with half the resolution, which helps the smoothness
    x = np.linspace(0, 1, n_op)
    A = f0 + (f1 - f0) * x

    iterations = len(max_offsets)
    max_num_of_intervals = n_op // (2 ** np.arange(iterations)[::-1])

    for i in range(iterations):
        interval_boundaries = np.sort(np.random.randint(1, n_op - 1, np.random.randint(2, max_num_of_intervals[i])))
        interval_lenghts = interval_boundaries[1:] - interval_boundaries[:-1]
        interval_offsets = np.random.uniform(-max_offsets[i], max_offsets[i], size=(len(interval_boundaries)-1,))
        offsets = np.repeat(interval_offsets, interval_lenghts)
        A[interval_boundaries[0]:interval_boundaries[-1]] += offsets

    indeksi0 = np.sort(np.random.permutation(np.arange(1, n_op - 1))[:np.random.randint(1, n_op // 8)])
    indeksi = np.concatenate(([0], indeksi0, [n_op - 1]))

    f = interp1d(x[indeksi], A[indeksi], kind='quadratic')

    return f(np.linspace(0, 1, n_points))


@njit
def theta_time_evolution(theta0, C, D=15 * 1e-6, dt=8 * 1e-8, num_timesteps=50000, nth_step_save=125):
    """
    Calculates time evolution (relaxation) of director profile (given by angle theta).

    :param theta0: Starting profile of theta (array of length N)
    :param C: Relaxation constant
    :param D: Thickness of the layer in meters
    :param dt: Timestep in seconds
    :param num_timesteps: Number of timesteps in the simulation
    :param nth_step_save: Save theta profile at every nth step (number of saves: M = num_timesteps // nth_step_save)
    :return: Time evolution of theta (array of dimensions (M, N))
    """

    N = len(theta0)
    dz = D / (N - 1)
    cnst = C * dt / (dz ** 2)

    if cnst > 0.5:
        raise ValueError("Iteration step too large, try smaller timestep or change other parameters.")

    thetas_out = np.zeros((num_timesteps // nth_step_save, N))
    theta1 = np.copy(theta0)
    for t in range(num_timesteps):
        theta1[1:N-1] += cnst * (theta1[2:N] - 2 * theta1[1:N-1] + theta1[:N-2])
        if t % nth_step_save == 0:
            thetas_out[t // nth_step_save, :] = theta1

    return thetas_out


@njit
def theta_time_evolution2(theta0, C, D=15 * 1e-6, dt=8 * 1e-8, num_timesteps=100000, nth_step_save=250):
    """
    Calculates time evolution (relaxation) of director profile (given by angle theta) for the case with both K1 and K3

    :param theta0: Starting profile of theta (array of length N)
    :param C: array containing both relaxation constants, K1/gamma and K3/gamma
    :param D: Thickness of the layer in meters
    :param dt: Timestep in seconds
    :param num_timesteps: Number of timesteps in the simulation
    :param nth_step_save: Save theta profile at every nth step (number of saves: M = num_timesteps // nth_step_save)
    :return: Time evolution of theta (array of dimensions (M, N))
    """

    N = len(theta0)
    dz = D / (N - 1)
    cnst = dt / (dz ** 2)

    if cnst * np.max(C) > 0.5:
        raise ValueError("Iteration step too large, try smaller timestep or change other parameters.")

    thetas_out = np.zeros((num_timesteps // nth_step_save, N))
    theta1 = np.copy(theta0)
    for t in range(num_timesteps):
        theta1[1:N-1] += cnst * ((theta1[2:N] - 2 * theta1[1:N-1] + theta1[:N-2]) *
                                 (C[0] * np.cos(theta1[1:N-1]) ** 2 + C[1] * np.sin(theta1[1:N-1]) ** 2) +
                                 1 / 8 * (C[1] - C[0]) * (theta1[2:N] - theta1[:N-2]) ** 2 * np.sin(2 * theta1[1:N-1]))
        if t % nth_step_save == 0:
            thetas_out[t // nth_step_save, :] = theta1

    return thetas_out


def intensity(theta_evolution, lbd=700 * 1e-9, D=15 * 1e-6, n_o0=1.53, n_e0=1.71):
    """
    Calculates time dependent intensity of transmitted linearly polarized light.

    :param theta_evolution: Time evolution of theta profile (array of dim (n_timesteps, n_zsteps))
    :param lbd: wavelength in meters
    :param D: thickness of the layer in meters
    :param n_o0: ordinary refractive index
    :param n_e0: extraordinary refractive index
    :return: intensity time dependence (array of length n_timesteps)
    """

    n_zsteps = theta_evolution.shape[1]
    h = D / (n_zsteps - 1)

    theta_evolution = (theta_evolution[:, :-1] + theta_evolution[:, 1:]) / 2

    n_e = (1 / ((np.cos(theta_evolution)) ** 2 / n_e0 ** 2 + (np.sin(theta_evolution)) ** 2 / n_o0 ** 2)) ** 0.5
    dPHI0 = np.sum((2 * np.pi / lbd) * h * (n_e - n_o0), axis=1)

    return (np.sin(dPHI0 / 2)) ** 2


def add_noise(data, amplitude, only_positive_allowed=True):
    """Adds white noise to data, with standard deviation 'amplitude'."""
    with_noise = data + np.random.normal(scale=amplitude, size=data.shape)
    if only_positive_allowed is True:
        with_noise[with_noise < 0] = 0
    return with_noise


def make_dir(directory):
    """Creates the desired directory. If it already exists, it is created with a superfix. Returns created dir name."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        return str(directory)
    else:
        i = 1
        while True:
            new_directory = directory + '_' + str(i)
            if not os.path.exists(new_directory):
                os.makedirs(new_directory)
                return str(new_directory)
            else:
                i += 1


class IntensityResult:
    def __init__(self, c, theta0, intsty):
        self.c = c
        self.theta0 = theta0
        self.intensity = intsty


class GenerateData:

    def __init__(self, D=15 * 1e-6, n_o0=1.53, n_e0=1.71, C_max=3. * 1e-8, no_constants=1):
        """
        Class in which data will be generated.
        :param D: thickness of the layer in meters
        :param n_o0: ordinary refractive index
        :param n_e0: extraordinary refractive index
        :param C_max: Maximal value for relaxation constants of interest
        :param no_constants: number of elastic constants in the model, must be 1 (for K1=K3) or 2
        """
        self.D = D
        self.n_o0 = n_o0
        self.n_e0 = n_e0
        self.C_max = C_max
        self.no_constants = no_constants
        if self.no_constants != 1 and self.no_constants != 2:
            raise ValueError("Parameter no_constants should be 1 or 2.")

        # we set the dimensions and values in the following arrays when method generate() is used
        self.lambdas = []
        self.all_cs = np.array([])
        self.all_funcs = np.array([])
        self.all_intensities = np.array([])

        # depending on number of constants, we set the relaxation function
        if self.no_constants == 1:
            self.relaxation = theta_time_evolution
        else:
            self.relaxation = theta_time_evolution2

        self.save_folder = None

    def generate(self, n: int, f0=0., f1: Union[float, None] = None,
                 zsteps=200, lbd: Union[int, float, list] = 700.,
                 dt=8 * 1e-8, num_timesteps=50000, nth_step_save=125, n_jobs=-1):
        """
        Method for generating data.
        :param n: number of data instances to generate
        :param f0: desired value of f(0)
        :param f1: desired value of f(1)
        :param zsteps: discretization number in z-dimension
        :param lbd: wavelengths of interest in nm
        :param dt: timestep
        :param num_timesteps: number of timesteps in the simulation
        :param nth_step_save: save theta profile at every nth step (number of saves: M = num_timesteps // nth_step_save)
        :param n_jobs: number of threads to use in the parallelized calculation
        """

        t0 = time.time()
        print(f'Generating {n} intensity functions...')

        if isinstance(lbd, float) or isinstance(lbd, int):
            lbd = [lbd]
        self.lambdas = lbd

        # generating random initial functions
        if self.no_constants == 1 and f1 is None:
            f1 = 0.
        if f1 is None:
            f1 = np.pi / 2

        fs = np.zeros((n, zsteps))
        for i in range(n):
            fs[i, :] = generate_function(zsteps, f0, f1)

        # generating random values for C
        if self.no_constants == 1:
            cs = self.C_max * np.random.uniform(0, 1, size=(n,))
        else:
            cs = self.C_max * np.random.uniform(0, 1, size=(n, 2))

        def instance(c, func):
            theta0 = np.copy(func)
            th = self.relaxation(func, c, D=self.D, dt=dt, num_timesteps=num_timesteps, nth_step_save=nth_step_save)
            intsty = [intensity(th, lbd=lam * 1e-9, D=self.D, n_o0=self.n_o0, n_e0=self.n_e0) for lam in lbd]
            return IntensityResult(c, theta0, intsty)

        # Parallel generation of individual instances for faster calculation
        y = Parallel(n_jobs=n_jobs)(delayed(instance)(c, f) for c, f in zip(cs, fs))

        if self.no_constants == 1:
            self.all_cs = np.zeros(n)
        else:
            self.all_cs = np.zeros((n, 2))

        self.all_intensities = np.zeros((len(lbd), n, num_timesteps // nth_step_save))
        self.all_funcs = np.zeros((n, zsteps))
        for i in range(n):
            self.all_cs[i] = y[i].c
            self.all_funcs[i] = y[i].theta0
            for j in range(len(lbd)):
                self.all_intensities[j, i] = y[i].intensity[j]

        t1 = time.time()
        print(f'Data generation finished in {np.round(t1 - t0, 2)} s.')

    def export(self, add_intensity_noise=0.0, path_to_save_folder=None, rewrite_existing=False):
        """Saves generated data in .npy format. Default location of the data folder is the current directory."""

        if path_to_save_folder is None:
            path_to_save_folder = os.path.dirname(os.path.realpath(__file__))

        if self.save_folder is None:
            if self.no_constants == 1:
                self.save_folder = path_to_save_folder + '/PodatkiK'
            else:
                self.save_folder = path_to_save_folder + '/PodatkiK13'
            if rewrite_existing is False:
                self.save_folder = make_dir(self.save_folder)

        np.save(self.save_folder + '/C_values.npy', self.all_cs)
        np.save(self.save_folder + '/theta0.npy', self.all_funcs)

        for i, lbd in enumerate(self.lambdas):
            intsty = add_noise(self.all_intensities[i], amplitude=add_intensity_noise, only_positive_allowed=True)
            np.save(self.save_folder + f'/intenziteta{int(lbd)}noise{int(1000 * add_intensity_noise)}.npy', intsty)


if __name__ == '__main__':

    obj = GenerateData(no_constants=2)
    obj.generate(100000, num_timesteps=100000, lbd=[500, 700, 900, 1000], nth_step_save=250, dt=8.*1e-8, f1=0)
    obj.export()
    # obj.export(add_intensity_noise=0.1)

    # obj = GenerateData(no_constants=1)
    # obj.generate(100000, num_timesteps=50000, lbd=[500, 700, 900, 1000], nth_step_save=125, dt=8.*1e-8, f1=np.pi/2)
    # obj.export()
    # obj.export(add_intensity_noise=0.1)
