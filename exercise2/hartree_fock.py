#! /usr/bin/env python3

"""
Hartree and Hartree-fock code for N-electron 1D harmonic quantum dot

- Related to Computational Physics 2

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import simps
# import h5py
import os

def hartree_potential(ns, x):
    """
    Hartree potential using Simpson integration
    :param ns: electron density
    :param x: grid
    :return: Vhartree: the hartree potential
    """

    Vhartree = 0.0 * ns
    for ix in range(len(x)):
        r = x[ix]
        f = 0.0 * x
        for ix2 in range(len(x)):
            rp = x[ix2]
            f[ix2] = ns[ix2] * ee_potential(r - rp)
        Vhartree[ix] = simps(f, x)
    return Vhartree

def calculate_exchange_potential(orbitals, i_orb, x):
    """
    Hartree-fock approximation replaces SIC with exchange potential
    :param density: electron density
    :param x: the grid
    :return: exchange potential
    """

    Vx = 0.0 * x
    for ix in range(len(x)):
        r = x[ix]
        f = 0.0 * x
        for ix2 in range(len(x)):
            rp = x[ix2]
            f[ix2] = density_HF(orbitals, ix, ix2, i_orb) * ee_potential(r - rp)
        Vx[ix] = simps(f, x)
    return -Vx

def ee_potential(x):
    """
    1D electron-electron interaction
    :param x: grid
    :return: The coulomb interaction between electrons given in the problem.
    """
    global ee_coef  # Global parameter containing information about electron-electron interaction.
    return ee_coef[0] / (np.sqrt(x ** 2 + ee_coef[1]))


def ext_potential(x, m=1.0, omega=1.0):
    """ 1D harmonic quantum dot """
    return 0.5 * m * omega ** 2 * x ** 2


def density(psis):
    """
    Calculate density using orbitals
    :param psis: vector containing the different orbitals as elements.
    :return: ns: electron density
    """
    ns = np.zeros((len(psis[0]),))
    for i in range(len(psis)):
        ns += abs(psis[i]) ** 2
    return ns

def density_HF(orbitals, ix, ix2, i_orb):
    """
    Calculates the electron density for the exchange potential in Hartree-Fock approximation.
    :param orbitals: orbital of the given electron.
    :return: fock_density: electron density
    """

    global spins
    fock_density = 0.0
    for j in range(len(orbitals)):  # Lecture 2 slides equation (6)
        if spins[i_orb] == spins[j]:
            fock_density += np.conj(orbitals[j][ix2])*orbitals[i_orb][ix2]*orbitals[j][ix]/orbitals[i_orb][ix]

    return fock_density



def initialize_density(x, dx, normalization=1):
    """ some kind of initial guess for the density """
    rho = np.exp(-x ** 2)
    A = simps(rho, x)
    return normalization / A * rho


def check_convergence(Vold, Vnew, threshold):
    """
    Checks if the effective potential has converged to the given threshold.
    :param Vold: Old effective potential
    :param Vnew: New effective potential
    :param threshold: the minimum difference between effective potentials we wish to achieve.
    :return: converged: True or False depending if the convergence is achieved.
    """
    difference_ = np.amax(abs(Vold - Vnew))
    print('  Convergence check:', difference_)
    converged = False
    if difference_ < threshold:
        converged = True
    return converged


def diagonal_energy(T, orbitals, x):
    """
    Calculate diagonal energy
    (using Simpson)
    """
    Tt = sp.csr_matrix(T)
    E_diag = 0.0

    for i in range(len(orbitals)):
        evec = orbitals[i]
        E_diag += simps(evec.conj() * Tt.dot(evec), x)
    return E_diag


def offdiag_potential_energy(orbitals, x):
    """
    Calculate off-diagonal energy
    (using Simpson)
    """
    U = 0.0
    for i in range(len(orbitals) - 1):
        for j in range(i + 1, len(orbitals)):
            fi = 0.0 * x
            for i1 in range(len(x)):
                fj = 0.0 * x
                for j1 in range(len(x)):
                    fj[j1] = abs(orbitals[i][i1]) ** 2 * abs(orbitals[j][j1]) ** 2 * ee_potential(x[i1] - x[j1])
                fi[i1] = simps(fj, x)
            U += simps(fi, x)
    return U

def fock_offdiag_potential_energy(orbitals, x):
    """
    Calculates the off-diagonal potential energy in the Hartree-fock approximation
    :param orbitals: orbitals of electrons
    :param x: the grid
    :return: fock potential energy
    """

    U_f = 0.0
    global spins

    for i in range(len(orbitals) - 1):  # Lecture 2 slides equation (10)
        for j in range(i + 1, len(orbitals)):
            if spins[i] == spins[j]:
                fi = 0.0 * x
                for i1 in range(len(x)):
                    fj = 0.0 * x
                    for j1 in range(len(x)):
                        fj[j1] = np.conj(orbitals[i][i1]) * np.conj(orbitals[j][j1]) * (orbitals[i][j1]) * orbitals[j][i1] * ee_potential(x[i1] - x[j1])
                    fi[i1] = simps(fj, x)
                U_f += simps(fi, x)
    return -U_f


def calculate_SIC(orbitals, x):
    """
    Self-interaction correction used in Hartree-approximation.
    :param orbitals: orbitals of the system
    :param x: the grid
    :return: Self-interaction correction
    """

    V_SIC = []
    for i in range(len(orbitals)):
        V_SIC.append(-hartree_potential(abs(orbitals[i]) ** 2, x))
    return V_SIC

def calculate_Vx(orbitals, x):
    """
    Exchange potential term used in the Hartree-Fock approximation
    :param orbitals: list of orbitals
    :param x: the grid
    :return: Vx exchange potential
    """
    Vx = []
    for i in range(len(orbitals)):
        Vx.append(calculate_exchange_potential(orbitals, i, x))
    return Vx

def normalize_orbital(evec, x):
    """ Normalize orbital properly """
    return evec / np.sqrt(simps(abs(evec) ** 2, x))


def kinetic_hamiltonian(x):
    grid_size = x.shape[0]
    dx = x[1] - x[0]
    dx2 = dx ** 2

    H0 = sp.diags(
        [
            -0.5 / dx2 * np.ones(grid_size - 1),
            1.0 / dx2 * np.ones(grid_size),
            -0.5 / dx2 * np.ones(grid_size - 1)
        ],
        [-1, 0, 1])
    return H0


def calculate_hartree(occ, new_run=True, fock=False):
    """
    Performs the SCF scheme and calls the other functions in order to perform it.
    :param occ: Indicates the State of the system. Meaning how the electrons are positioned within the system.
    :param new_run: Tells the function if we wish to continue the calculation from previous run (saved text file).
    :param fock: Tells the function if we wish to calculate hartree approximation or hartree-fock approximation.
    :return: values needed for plotting the figure. e.g. grid, electron density and energetics.
    """
    # number of electrons
    N_e = len(occ)

    # grid
    x = np.linspace(-4, 4, 120)
    # threshold
    threshold = 1.0e-4
    # mixing value
    mix_alpha = 0.2
    # maximum number of iterations
    maxiters = 100

    dx = x[1] - x[0]
    T = kinetic_hamiltonian(x)
    Vext = ext_potential(x)

    # READ in density / orbitals / etc. (related to problems 3 and 4)
    if os.path.isfile('density.txt') and not new_run:
        ns, orbitals = load_data(N_e)
        print('\nCalculating initial state')
        Vhartree = hartree_potential(ns, x)
        if fock is True:
            VSIC = calculate_Vx(orbitals, x)
        else:
            VSIC = calculate_SIC(orbitals, x)

    else:
        ns = initialize_density(x, dx, N_e)
        print('\nCalculating initial state')
        Vhartree = hartree_potential(ns, x)
        VSIC = []
        for i in range(N_e):
            VSIC.append(ns * 0.0)

    print('Density integral        ', simps(ns, x))
    print(' -- should be close to  ', N_e)

    Veff = sp.diags(Vext + Vhartree, 0)
    H = T + Veff
    for i in range(maxiters):  # SCF scheme
        print('\n\nIteration #{0}'.format(i))
        orbitals = []
        for i in range(N_e):
            print('  Calculating orbitals for electron ', i + 1)
            eigs, evecs = sla.eigs(H + sp.diags(VSIC[i], 0), k=N_e, which='SR')
            eigs = np.real(eigs)
            evecs = np.real(evecs)
            print('    eigenvalues', eigs)
            evecs[:, occ[i]] = normalize_orbital(evecs[:, occ[i]], x)
            orbitals.append(evecs[:, occ[i]])
        Veff_old = 1.0 * Veff
        ns = density(orbitals)
        Vhartree = hartree_potential(ns, x)
        if fock is True:
            VSIC = calculate_Vx(orbitals,x)
        else:
            VSIC = calculate_SIC(orbitals, x)
        Veff_new = sp.diags(Vext + Vhartree, 0)
        if check_convergence(Veff_old, Veff_new, threshold):
            break
        else:
            """ Mixing the potential, mix_alpha is the mixing parameter """
            Veff = mix_alpha * Veff_old + (1 - mix_alpha) * Veff_new
            H = T + Veff

    print('\n\n')
    off = offdiag_potential_energy(orbitals, x)
    E_kin = diagonal_energy(T, orbitals, x)
    E_pot = diagonal_energy(sp.diags(Vext, 0), orbitals, x) + off
    if fock is True:
        E_pot += fock_offdiag_potential_energy(orbitals, x)
    E_tot = E_kin + E_pot
    print('Total energy     ', E_tot)
    print('Kinetic energy   ', E_kin)
    print('Potential energy ', E_pot)
    print('\nDensity integral ', simps(ns, x))

    # WRITE OUT density / orbitals / energetics / etc. (related to problems 3 and 4)

    save_data(ns, orbitals)

    return x, ns, E_kin, E_pot, E_tot, N_e


def save_data(density, orbitals):
    """
    Saves the densities and orbitals into a file.
    :param density: electron density
    :param orbitals: orbitals
    :return:
    """
    np.savetxt('density.txt', density)
    for i in range(len(orbitals)):
        np.savetxt('orbital_{}.txt'.format(i), orbitals[i])


def load_data(Norbs):
    """
    Loads the text file in order to continue calculations.
    :param Norbs:
    :return:
    """
    density = np.loadtxt('density.txt')
    orbitals = []
    for i in range(Norbs):
        orbitals.append(np.loadtxt('orbital_{}.txt'.format(i)))
    return density, orbitals


def plot_figure(x, ns, E_kin, E_pot, E_tot, N_e):
    """
    Plot the figure of electron density in relation to x alongside of energetics. Used when single figure is wanted.
    :param x: grid
    :param ns:  electron density
    :param E_kin:   Kinetic energy
    :param E_pot:   Potential energy
    :param E_tot:   Total energy
    :param N_e:     Number of electrons
    :return:
    """

    text1 = "Total energy: " + str(round(E_tot, 5)) + "\n" + 'Kinetic energy: ' + str(round(E_kin, 5)) + '\n' + \
            'Potential energy: ' + str(round(E_pot, 5))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    plt.figure()
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.handlelength'] = 2
    plt.rcParams['legend.numpoints'] = 1
    plt.plot(x, abs(ns))
    plt.xlabel(r'$x$ (a.u.)')
    plt.ylabel(r'$n(x)$ (1/a.u.)')
    plt.title(r'$N$-electron density for $N={0}$'.format(N_e))
    plt.text(-2, 0.2, text1, fontsize=12, bbox=props)
    plt.show()


def compare_hartree_fock(x1, x2, ns1, ns2, E_kin1, E_kin2, E_pot1, E_pot2, E_tot1, E_tot2, N_e1, N_e2):
    """
    Plots the two figures and compares the Hartree and Hartree-fock approximations and their energetics
    :param x1:  grid of hartree
    :param x2:  grid of hartree-fock
    :param ns1: electron density of hartree
    :param ns2: electron density of hartree-fock
    :param E_kin1:  Kinetic energy of hartree
    :param E_kin2:  Kinetic energy of hartree-fock
    :param E_pot1:  Potential energy of hartree
    :param E_pot2:  Potential energy of hartree-fock
    :param E_tot1:  Total energy of hartree
    :param E_tot2:  Total energy of hartree-fock
    :param N_e1:    Number of electrons hartree
    :param N_e2:    Number of electrons hartree-fock
    :return:
    """

    E_kin_diff = abs(E_kin1 - E_kin2)
    E_pot_diff = abs(E_pot1 - E_pot2)
    E_tot_diff = abs(E_tot1 - E_tot2)

    print()
    print('Hartree and Hartree-fock')
    print('Differences in energies:')
    print('N = 6 and S = 0: ')
    print()
    print('E_kin: ', E_kin_diff)
    print('E_pot: ', E_pot_diff)
    print('E_tot: ', E_tot_diff)

    text1 = "Hartree approximation:" + "\n" + "Total energy: " + str(round(E_tot1, 5)) + "\n" + 'Kinetic energy: ' + str(round(E_kin1, 5)) + '\n' + \
            'Potential energy: ' + str(round(E_pot1, 5)) + '\n' + 'Density integral: ' + str(round(simps(ns1, x1)))
    props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    text2 = "Hartree-fock approximation:" + "\n" + "Total energy: " + str(round(E_tot2, 5)) + "\n" + 'Kinetic energy: ' + str(round(E_kin2, 5)) + '\n' + \
            'Potential energy: ' + str(round(E_pot2, 5)) + '\n' + 'Density integral: ' + str(round(simps(ns2, x2)))
    props2 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    fig = plt.figure()
    f1 = fig.add_subplot(121)
    f2 = fig.add_subplot(122)

    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.handlelength'] = 2
    plt.rcParams['legend.numpoints'] = 1
    plt.xlabel(r'$x$ (a.u.)')
    plt.ylabel(r'$n(x)$ (1/a.u.)')

    f1.plot(x1, abs(ns1))
    f1.set_title(r'$N$-electron density for $N={0}$, (S = 0)'.format(N_e1))
    f1.text(-2, 0.2, text1, fontsize=12, bbox=props1)

    f2.plot(x2, abs(ns2))
    f2.set_title(r'$N$-electron density for $N={0}$, (S = 0)'.format(N_e2))
    f2.text(-2, 0.2, text2, fontsize=12, bbox=props2)
    plt.show()


def main():
    """
    Main function that gives the occ values to the calculate_hartree and calculate_HF functions.
    :return:
    """
    global ee_coef
    # e-e potential parameters [strength, smoothness]
    ee_coef = [1.0, 1.0]

    # 1D occupations each orbital can have max 2, i.e., spin up and spin down
    # e.g., occ = [0,0,1,1] means lowest up and down, next lowest up and down
    #       occ = [0,1,2,3] means all spin up

    occ = [0, 0, 1, 1, 2, 2]  # test case (N = 6 and S = 0)

    global spins  # Declarinbg the spins for the hartree-fock approximation
    spins = [0, 1, 0, 1, 0, 1]

    # Comparing Hartree and Hartree-Fock approximations
    x1, ns1, E_kin1, E_pot1, E_tot1, N_e1 = calculate_hartree(occ, True, False)  # Hartree approximation
    x2, ns2, E_kin2, E_pot2, E_tot2, N_e2 = calculate_hartree(occ, True, True)  # Hartree-Fock approximation

    compare_hartree_fock(x1, x2, ns1, ns2, E_kin1, E_kin2, E_pot1, E_pot2, E_tot1, E_tot2, N_e1, N_e2)


if __name__ == "__main__":
    main()
