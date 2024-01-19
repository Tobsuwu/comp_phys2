"""
Simulation of Ising model of a ferromagnet in regular 2D grid with Monte Carlo.
Heavily related to the model introduced in exercise5.

Perform simulations in two ways:
a) Using only the nearest neighbor interactions with J1 = 4.
b) Using both the nearest neighbor and next nearest neighbor interactions, i.e., J1 = 4 and J2 = 1.

"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from matplotlib.pyplot import *

class Walker:
    """ An object that describes the properties of the system and a single point in it. In this case can be loosely
    thought as a lattice ion with few extra attributes """

    def __init__(self,*args,**kwargs):
        self.spin = kwargs['spin']
        self.nearest_neighbors = kwargs['nn']
        self.sys_dim = kwargs['dim']
        self.coords = kwargs['coords']

    def w_copy(self):
        return Walker(spin=self.spin.copy(),
                      nn=self.nearest_neighbors.copy(),
                      dim=self.sys_dim,
                      coords=self.coords.copy())

def Magnetization(Walkers):
    """ Calculates the "measured" magnetization. Magnetic moment of the microstate is the sum of the spin values."""
    m = 0.0
    for i in range(len(Walkers)):
        m += Walkers[i].spin
    return m

def Energy(Walkers):
    """
    Calculates the simplest Ising model energy assuming interaction between nearest neighbours
    and next nearest neighbours depending on the given value.
    :param Walkers: List of walkers
    :return:
    """
    global next_nearest
    E1 = 0.0  # Energy with the nearest neighbour interaction
    E2 = 0.0  # Energy with the next nearest neighbour interaction
    J1 = 4.0  # given in units of k_B
    J2 = 1

    for k in range(len(Walkers)):  # Find all the nearest neighbours for all the walkers.
        for i in range(len(Walkers[k].nearest_neighbors)):
            j = Walkers[k].nearest_neighbors[i]
            E1 += -J1*Walkers[k].spin*Walkers[j].spin
        if next_nearest:
            for nnn in range(len(Walkers[j].nearest_neighbors)):  # nnn is the next nearest neighbour
                n = Walkers[j].nearest_neighbors[nnn]
                E2 += -J2*Walkers[k].spin*Walkers[n].spin

    E_final = E1/2 + E2/2  # Double counting (repentance of nearest neighbour interaction) taken into account
    return E_final

def site_Energy(Walkers,Walker):
    """
    Calculates the energy of the system on a specific spin state s.
    :param Walkers: List of walkers
    :param Walker: The walker that determines the specific spin state of the system.
    :return:
    """

    global next_nearest
    E1 = 0.0
    E2 = 0.0
    J1 = 4.0  # given in units of k_B
    J2 = 1.0

    for k in range(len(Walker.nearest_neighbors)):
        j = Walker.nearest_neighbors[k]
        E1 += -J1*Walker.spin*Walkers[j].spin
        if next_nearest:
            for nnn in range(len(Walkers[j].nearest_neighbors)):
                n = Walkers[j].nearest_neighbors[nnn]
                E2 += -J2*Walker.spin*Walkers[n].spin

    return E1 + E2

def ising(Nblocks,Niters,Walkers,beta):
    """
    Performs the simulation of the ising model
    :param Nblocks: number of blocks in the run
    :param Niters: number of iterations inside the block
    :param Walkers: list of walkers
    :param beta: 1/T
    :return:
    """
    M = len(Walkers)
    Eb = zeros((Nblocks,))
    Eb2 = zeros((Nblocks,))
    Mb = zeros((Nblocks,))
    Mb2 = zeros((Nblocks,))
    Accept=zeros((Nblocks,))
    AccCount=zeros((Nblocks,))

    obs_interval = 5
    for i in range(Nblocks):
        EbCount = 0
        for j in range(Niters):
            site = int(random.rand()*M)

            s_old = 1.0*Walkers[site].spin
 
            E_old = site_Energy(Walkers,Walkers[site])

            # ADD selection of new spin to variable s_new
            # Spin is either 1/2 or -1/2. Roll the dice
            rand_num = random.rand()
            if rand_num <= 1/2:
                s_new = 1/2
            else:
                s_new = -1/2

            Walkers[site].spin = 1.0*s_new

            E_new = site_Energy(Walkers,Walkers[site])

            # ADD Metropolis Monte Carlo
            # Measure of acceptance, lecture 5 slide 15. Uniform distribution so the S(s'->s)/S(s->s') = 1
            q_s_old_to_s_new = exp(-beta*(E_new-E_old))
            A_s_old_to_s_new = min(1.0, q_s_old_to_s_new)
            if A_s_old_to_s_new > random.rand():
                Accept[i] += 1.0
            else:
                Walkers[site].spin = 1.0*s_old
            AccCount[i] += 1

            if j % obs_interval == 0:
                E_tot = Energy(Walkers)  # energy per spin
                Eb[i] += E_tot
                Eb2[i] += E_tot**2
                M_tot = Magnetization(Walkers)  # magnetization per spin
                Mb[i] += M_tot
                Mb2[i] += M_tot**2
                EbCount += 1
            
        Eb[i] /= EbCount
        Eb2[i] /= EbCount
        Mb[i] /= EbCount
        Mb2[i] /= EbCount
        Accept[i] /= AccCount[i]
        print('Block {0}/{1}'.format(i+1,Nblocks))
        print('    E   = {0:.5f}'.format(Eb[i]))
        print('    Acc = {0:.5f}'.format(Accept[i]))
        print('    M   = {0:.5f}'.format(Mb[i]))

    return Walkers, Eb, Eb2, Mb, Mb2, Accept


def nn_or_nnn(Walkers, Nblocks, Niters, grid_size):
    """
    Calculates the means of the observables and plots them as a function of temperature. Figure in the repo is done
    with 20 points in T linspace. Runtime is manageable (takes a couple of minutes).
    :param Walkers: list of walkers
    :param Nblocks: number of blocks used in the ising model
    :param Niters: number of iterations inside the block
    :param grid_size: size of our grid (lattice)
    :return:
    """

    T = linspace(0.5, 10, 20)  # Temperature (K)

    # Initializes the means
    Eb_T = zeros_like(T)
    Mb_T = zeros_like(T)
    Cv_T = zeros_like(T)
    suscep_T = zeros_like(T)
    for i in range(len(T)):  # Go through the temperature grid and save means of the observables corresponding to the T.
        Walkers, Eb, Eb2, Mb, Mb2, Acc = ising(Nblocks, Niters, Walkers, 1.0 / T[i])  # beta 1/T
        Eb_T[i] = mean(Eb / grid_size)
        Mb_T[i] = mean(Mb / grid_size)
        Cv_T[i] = (mean(Eb2) - mean(Eb) ** 2) / (T[i] ** 2 * grid_size)
        suscep_T[i] = (mean(Mb2) - mean(Mb) ** 2) / (T[i] * grid_size)

    # plot the figures
    fig = figure()
    f1 = fig.add_subplot(221)
    f2 = fig.add_subplot(222)
    f3 = fig.add_subplot(223)
    f4 = fig.add_subplot(224)

    f1.plot(T, Eb_T, linestyle='--', marker='o', color='b', label='line with marker')
    f1.set_xlabel('Temperature (K)')
    f1.set_ylabel('Energy per spin')

    f2.plot(T, Mb_T, linestyle='--', marker='o', color='b', label='line with marker')
    f2.set_xlabel('Temperature (K)')
    f2.set_ylabel('Magnetization per spin')

    f3.plot(T, Cv_T, linestyle='--', marker='o', color='b', label='line with marker')
    f3.set_xlabel('Temperature (K)')
    f3.set_ylabel('Heat capacity per spin')

    f4.plot(T, suscep_T, linestyle='--', marker='o', color='b', label='line with marker')
    f4.set_xlabel('Temperature (K)')
    f4.set_ylabel('Susceptibility per spin')
    show()

    """ Judging from the figures, it looks like the transition temperature is roughly in range of [2.2 K, 2.5 K] """

def finite_size(Walkers, Nblocks, Niters, grid_size):

    T = linspace(0.5, 6, 20)  # Temperature (K)

    # Initializes the means
    Eb_T = zeros_like(T)
    for i in range(len(T)):  # Go through the temperature grid and save means of the observables corresponding to the T.
        Walkers, Eb, Eb2, Mb, Mb2, Acc = ising(Nblocks, Niters, Walkers, 1.0 / T[i])  # beta 1/T
        Eb_T[i] = mean(Eb / grid_size)

    return Eb_T

def plot_fz(E_fz):

    T = np.linspace(0.5, 6, 20)
    plt.figure()
    plt.plot(T, E_fz[0], label='N = 4')
    plt.plot(T, E_fz[1], label='N = 8')
    plt.plot(T, E_fz[2], label='N = 16')
    plt.plot(T, E_fz[3], label='N = 32')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Energy per spin')
    plt.title('Energy per spin in relation to temperature with different grid sizes N')
    plt.legend()
    plt.show()

def run_main(grid_side):
    """
    Main function that runs the program.
    :return:
    """
    Walkers=[]

    dim = 2
    grid_size = grid_side**dim
    
    # Ising model nearest neighbors only
    mapping = zeros((grid_side,grid_side),dtype=int) # mapping
    inv_map = []  # inverse mapping
    ii = 0
    for i in range(grid_side):
        for j in range(grid_side):
            mapping[i,j]=ii
            inv_map.append([i,j])
            ii += 1

    # Allocates nearest neighbours for all the walkers.
    for i in range(grid_side):
        for j in range(grid_side):
            j1=mapping[i,(j-1) % grid_side]
            j2=mapping[i,(j+1) % grid_side]
            i1=mapping[(i-1) % grid_side,j]
            i2=mapping[(i+1) % grid_side,j]
            Walkers.append(Walker(spin=0.5,
                                  nn=[j1,j2,i1,i2],
                                  dim=dim,
                                  coords = [i,j]))

    Nblocks = 200
    Niters = 1000  # THIS WAS 1000


    """
    Notice: Energy is measured in units of k_B, which is why
            beta = 1/T instead of 1/(k_B T)
    """

    # Energetics with only nearest neighbour interactions or next nearest and nearest neighbour interactions.
    global next_nearest
    next_nearest = False

    # Used to compare the energetics with nearest neighbour interactions and next nearest interactions.
    nn_or_nnn(Walkers, Nblocks, Niters, grid_size)

    # Used to compare the finite size effects in case of nearest neighbour interactions
    #Energy_per_spin_T = finite_size(Walkers, Nblocks, Niters, grid_size)
    #return Energy_per_spin_T

def main():

    # Used to compare the energetics with nearest neighbour interactions and next nearest interactions.
    grid_side = 10
    run_main(grid_side)

    # Used to compare the finite size effects in case of nearest neighbour interactions
    #grid_side = [4, 8, 16, 32]
    #E_fz = []
    #for i in range(len(grid_side)):
        #E = run_main(grid_side[i])
        #E_fz.append(E)

    #plot_fz(E_fz)


if __name__=="__main__":
    main()
        
