#! /usr/bin/env python3

"""
Hartree code for N-electron 1D harmonic quantum dot

- Related to Computational Physics 2
- Test case should give values very close to:

    Total energy      11.502873299221452
    Kinetic energy    3.6221136067112054
    Potential energy  7.880759692510247

    Density integral  4.0

- Job description: 
  -- Problem 2: add/fill needed functions, details and especially comments
  -- Problem 3: Modify parameters for calculating a different system
                as described in the problem setup
  -- Problem 4: Include input and output as text file
  -- Problem 5: Include input and output as DHF5 file

- Notice: In order to get the code running you need to add proper code sections 
  to places where it reads #FILL#
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import simps
#import h5py
import os

def hartree_potential(ns,x):
    """
    Hartree potential using Simpson integration
    :param ns: electron density
    :param x: grid
    :return: Vhartree: the hartree potential
    """

    Vhartree = 0.0*ns
    for ix in range(len(x)):
        r = x[ix]
        f = 0.0*x
        for ix2 in range(len(x)):
            rp = x[ix2]
            f[ix2] = ns[ix2]*ee_potential(r-rp)
        Vhartree[ix] = simps(f, x)
    return Vhartree

def ee_potential(x):
    """
    1D electron-electron interaction
    :param x: grid
    :return: The coulomb interaction between electrons given in the problem.
    """
    global ee_coef  # Global parameter containing information about electron-electron interaction.
    return ee_coef[0]/(np.sqrt(x**2 + ee_coef[1]))

def ext_potential(x,m=1.0,omega=1.0):
    """ 1D harmonic quantum dot """
    return 0.5*m*omega**2*x**2

def density(psis):
    """
    Calculate density using orbitals
    :param psis: vector containing the different orbitals as elements.
    :return: ns: electron density
    """
    ns=np.zeros((len(psis[0]),))
    for i in range(len(psis)):
        ns += abs(psis[i])**2
    return ns
    
def initialize_density(x,dx,normalization=1):
    """ some kind of initial guess for the density """
    rho = np.exp(-x**2)
    A = simps(rho,x)
    return normalization/A*rho

def check_convergence(Vold,Vnew,threshold):
    """
    Checks if the effective potential has converged to the given threshold.
    :param Vold: Old effective potential
    :param Vnew: New effective potential
    :param threshold: the minimum difference between effective potentials we wish to achieve.
    :return: converged: True or False depending if the convergence is achieved.
    """
    difference_ = np.amax(abs(Vold-Vnew))
    print('  Convergence check:', difference_)
    converged = False
    if difference_ <threshold:
        converged = True
    return converged

def diagonal_energy(T,orbitals,x):
    """ 
    Calculate diagonal energy
    (using Simpson)
    """
    Tt = sp.csr_matrix(T)
    E_diag = 0.0
    
    for i in range(len(orbitals)):
        evec = orbitals[i]
        E_diag += simps(evec.conj()*Tt.dot(evec),x)
    return E_diag

def offdiag_potential_energy(orbitals,x):
    """ 
    Calculate off-diagonal energy
    (using Simpson)
    """
    U = 0.0
    for i in range(len(orbitals)-1):
        for j in range(i+1,len(orbitals)):
            fi = 0.0*x
            for i1 in range(len(x)):
                fj = 0.0*x
                for j1 in range(len(x)):
                    fj[j1] = abs(orbitals[i][i1])**2*abs(orbitals[j][j1])**2*ee_potential(x[i1]-x[j1])
                fi[i1] = simps(fj,x)
            U += simps(fi,x)
    return U

def save_ns_in_ascii(ns,filename):
    s = ns.shape
    f = open(filename+'.txt','w')
    for ix in range(s[0]):
        f.write('{0:12.8f}\n'.format(ns[ix]))
    f.close()
    f = open(filename+'_shape.txt','w')
    f.write('{0:5}'.format(s[0]))
    f.close()
    
def load_ns_from_ascii(filename):
    f = open(filename+'_shape.txt','r')
    for line in f:
        s = np.array(line.split(),dtype=int)
    f.close()
    ns = np.zeros((s[0],))
    d = np.loadtxt(filename+'.txt')
    k = 0
    for ix in range(s[0]):
        ns[ix] = d[k]
        k += 1
    return ns


def save_data_to_hdf5_file(fname,orbitals,density,N_e,occ,grid,ee_coefs):
    return

def calculate_SIC(orbitals,x):
    V_SIC = []
    for i in range(len(orbitals)):
        V_SIC.append(-hartree_potential(abs(orbitals[i])**2,x))
    return V_SIC
            
def normalize_orbital(evec,x):
    """ Normalize orbital properly """
    return evec/np.sqrt(simps(abs(evec)**2, x))

def kinetic_hamiltonian(x):
    grid_size = x.shape[0]
    dx = x[1] - x[0]
    dx2 = dx**2
    
    H0 = sp.diags(
        [
            -0.5 / dx2 * np.ones(grid_size - 1),
            1.0 / dx2 * np.ones(grid_size),
            -0.5 / dx2 * np.ones(grid_size - 1)
        ],
        [-1, 0, 1])
    return H0

def fake_main(occ, new_run = True):
    """
    The "original" main function. Modified to make the code run more conveniently for the problems. Performs the
    SCF scheme and calls the other functions in order to perform it.
    :param occ: Indicates the State of the system. Meaning how the electrons are positioned within the system.
    :return: values needed for plotting the figure. e.g. grid, electron density and energetics.
    """
    # number of electrons
    N_e = len(occ)

    # grid
    x = np.linspace(-4,4,120)
    # threshold
    threshold = 1.0e-4
    # mixing value
    mix_alpha = 0.2
    # maximum number of iterations
    maxiters = 100

    dx = x[1]-x[0]
    T = kinetic_hamiltonian(x)
    Vext = ext_potential(x)

    # READ in density / orbitals / etc. (related to problems 3 and 4)
    if os.path.isfile('density.txt') and not new_run:
        ns, orbitals = load_data(N_e)
        print('\nCalculating initial state')
        Vhartree = hartree_potential(ns, x)
        VSIC = calculate_SIC(orbitals, x)
    else:
        ns = initialize_density(x,dx,N_e)
        print('\nCalculating initial state')
        Vhartree = hartree_potential(ns, x)
        VSIC = []
        for i in range(N_e):
            VSIC.append(ns * 0.0)

    print('Density integral        ', simps(ns,x))
    print(' -- should be close to  ', N_e)

    Veff = sp.diags(Vext+Vhartree,0)
    H = T+Veff
    for i in range(maxiters):  # SCF scheme
        print('\n\nIteration #{0}'.format(i))
        orbitals = []
        for i in range(N_e):
            print('  Calculating orbitals for electron ', i+1)
            eigs, evecs = sla.eigs(H+sp.diags(VSIC[i],0), k=N_e, which='SR')
            eigs = np.real(eigs)
            evecs = np.real(evecs)
            print('    eigenvalues', eigs)
            evecs[:,occ[i]] = normalize_orbital(evecs[:,occ[i]],x)
            orbitals.append(evecs[:,occ[i]])
        Veff_old = 1.0*Veff
        ns = density(orbitals)
        Vhartree = hartree_potential(ns,x)
        VSIC = calculate_SIC(orbitals,x)
        Veff_new = sp.diags(Vext+Vhartree,0)
        if check_convergence(Veff_old,Veff_new,threshold):
            break
        else:
            """ Mixing the potential, mix_alpha is the mixing parameter """
            Veff = mix_alpha*Veff_old + (1-mix_alpha)*Veff_new
            H = T+Veff

    print('\n\n')
    off = offdiag_potential_energy(orbitals,x)
    E_kin = diagonal_energy(T,orbitals,x)
    E_pot = diagonal_energy(sp.diags(Vext,0),orbitals,x) + off
    E_tot = E_kin + E_pot
    print('Total energy     ', E_tot)
    print('Kinetic energy   ', E_kin)
    print('Potential energy ', E_pot) 
    print('\nDensity integral ', simps(ns,x))

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

def plot_figure_p2(x, ns, E_kin, E_pot, E_tot, N_e):
    """
    Plot the figure of electron density in relation to x alongside of energetics.
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
    plt.plot(x,abs(ns))
    plt.xlabel(r'$x$ (a.u.)')
    plt.ylabel(r'$n(x)$ (1/a.u.)')
    plt.title(r'$N$-electron density for $N={0}$'.format(N_e))
    plt.text(-2, 0.2, text1, fontsize=12, bbox=props)
    plt.show()

def plot_figure_p3(x1, x2, ns1, ns2, E_kin1, E_kin2, E_pot1, E_pot2, E_tot1, E_tot2, N_e1, N_e2):
    """
    Plots the two figures of S=0 and S=4 side by side with corresponding energetics.
    :param x1:  grid of S=0
    :param x2:  grid of S=4
    :param ns1: electron density of S=0
    :param ns2: electron density of S=4
    :param E_kin1:  Kinetic energy of S=0
    :param E_kin2:  Kinetic energy of S=4
    :param E_pot1:  Potential energy of S=0
    :param E_pot2:  Potential energy of S=4
    :param E_tot1:  Total energy of S=0
    :param E_tot2:  Total energy of S=4
    :param N_e1:    Number of electrons S=0
    :param N_e2:    Number of electrons S=4
    :return:
    """

    text1 = "Total energy: " + str(round(E_tot1, 5)) + "\n" + 'Kinetic energy: ' + str(round(E_kin1, 5)) + '\n' + \
            'Potential energy: ' + str(round(E_pot1, 5)) + '\n' + 'Density integral: ' + str(round(simps(ns1, x1)))
    props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    text2 = "Total energy: " + str(round(E_tot2, 5)) + "\n" + 'Kinetic energy: ' + str(round(E_kin2, 5)) + '\n' + \
            'Potential energy: ' + str(round(E_pot2, 5)) + '\n' +'Density integral: ' + str(round(simps(ns2, x2)))
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
    f2.set_title(r'$N$-electron density for $N={0}$, (S = 4)'.format(N_e2))
    f2.text(-2, 0.2, text2, fontsize=12, bbox=props2)
    plt.show()

def problem3():
    """
    Code for problem3
    :return:
    """

    # 1D occupations each orbital can have max 2, i.e., spin up and spin down
    # e.g., occ = [0,0,1,1] means lowest up and down, next lowest up and down
    #       occ = [0,1,2,3] means all spin up

    occ3_1 = [0, 0, 1, 1, 2, 2, 3, 3]  # test case P3, Hartree ground state S = 0
    occ3_2 = [0, 1, 2, 3, 4, 5, 6, 7]  # test case P3, Hartree excited state S = 4

    x1, ns1, E_kin1, E_pot1, E_tot1, N_e1 = fake_main(occ3_1)
    x2, ns2, E_kin2, E_pot2, E_tot2, N_e2 = fake_main(occ3_2)

    plot_figure_p3(x1, x2, ns1, ns2, E_kin1, E_kin2, E_pot1, E_pot2, E_tot1, E_tot2, N_e1, N_e2)

    # Non-interacting case -> Coulomb interaction between the electrons is zero
    # Only one SCF iteration is needed for the non-interacting case.

    global ee_coef
    ee_coef = [0, 1]
    x01, ns01, E_kin01, E_pot01, E_tot01, N_e01 = fake_main(occ3_1)
    x02, ns02, E_kin02, E_pot02, E_tot02, N_e02 = fake_main(occ3_2)

    plot_figure_p3(x01, x02, ns01, ns02, E_kin01, E_kin02, E_pot01, E_pot02, E_tot01, E_tot02, N_e01, N_e02)

    E_kin_diff_00 = abs(E_kin01 - E_kin1)
    E_kin_diff_04 = abs(E_kin02 - E_kin2)
    E_pot_diff_00 = abs(E_pot01 - E_pot1)
    E_pot_diff_04 = abs(E_pot02 - E_pot2)
    E_tot_diff_00 = abs(E_tot01 - E_tot1)
    E_tot_diff_04 = abs(E_tot02 - E_tot2)

    print()
    print('Differences in energies:')
    print()
    print('Non interacting and S = 0: ')
    print('E_kin: ', E_kin_diff_00)
    print('E_pot: ', E_pot_diff_00)
    print('E_tot: ', E_tot_diff_00)
    print()
    print('Non interacting and S = 4: ')
    print('E_kin: ', E_kin_diff_04)
    print('E_pot: ', E_pot_diff_04)
    print('E_tot: ', E_tot_diff_04)

def main():  # True main function that gives the occ values to the fake_main function (original main function)

    global ee_coef
    # e-e potential parameters [strength, smoothness]
    ee_coef = [1.0, 1.0]

    # 1D occupations each orbital can have max 2, i.e., spin up and spin down
    # e.g., occ = [0,0,1,1] means lowest up and down, next lowest up and down
    #       occ = [0,1,2,3] means all spin up

    occ2 = [0, 1, 2, 3]  # test case (i.e., P2)

    # Problem 2

    x, ns, E_kin, E_pot, E_tot, N_e = fake_main(occ2)
    plot_figure_p2(x, ns, E_kin, E_pot, E_tot, N_e)

    # Problem 3

    problem3()

if __name__=="__main__":
    main()
