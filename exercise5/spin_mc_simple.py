"""
Simple Monte Carlo for Ising model

Related to course Computational Physics 2 at TAU

Problem 1:
- Make the code work, that is, include code to where it reads "# ADD"
- Comment the parts with "# ADD" and make any additional comments you 
  think could be useful for yourself later.
- Follow the assignment from ex5.pdf.

Problem 2:
- Add observables: heat capacity, magnetization, magnetic susceptibility
- Follow the assignment from ex5.pdf.

Problem 3:
- Look at the temperature effects and locate the phase transition temperature.
- Follow the assignment from ex5.pdf.

"""
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
    Calculates the simplest Ising model energy assuming interactin only between nearest neighbours.
    :param Walkers: List of walkers
    :return:
    """
    E = 0.0
    J = 4.0 # given in units of k_B
    # ADD calculation of energy
    for k in range(len(Walkers)):  # Find all the nearest neighbours for all the walkers.
        for i in range(len(Walkers[k].nearest_neighbors)):
            j = Walkers[k].nearest_neighbors[i]
            E += -J*Walkers[k].spin*Walkers[j].spin
    E_final = E/2  # Double counting (repentance of nearest neighbour interaction) taken into account
    return E_final

def site_Energy(Walkers,Walker):
    """
    Calculates the energy of the system on a specific spin state s.
    :param Walkers: List of walkers
    :param Walker: The walker that determines the specific spin state of the system.
    :return:
    """
    E = 0.0
    J = 4.0 # given in units of k_B
    for k in range(len(Walker.nearest_neighbors)):
        j = Walker.nearest_neighbors[k]
        E += -J*Walker.spin*Walkers[j].spin
    return E

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

def problem2(Walkers, Nblocks, Niters, grid_size):
    """
    Function that runs the problem2 of the exercise5 set.
    :param Walkers: List of walkers
    :param Nblock: Number of blocks used in the run
    :param Niters: number of iterations inside the block
    :param grid_size: size of the grid (lattice)
    :return:
    """

    eq = 20  # equilibration "time"
    T = 3.0
    beta = 1.0 / T

    Walkers, Eb, Eb2, Mb, Mb2, Accept = ising(Nblocks, Niters, Walkers, beta)

    Eb = Eb[eq:]
    Eb2 = Eb2[eq:]
    Mb = Mb[eq:]
    Mb2 = Mb2[eq:]

    Cv = (mean(Eb2) - mean(Eb) ** 2) / (T ** 2 * grid_size)
    suscep = (mean(Mb2) - mean(Mb) ** 2) / (T * grid_size)

    print('Ising total energy: {0:.5f} +/- {1:0.5f}'.format(mean(Eb / grid_size),
                                                            std(Eb / grid_size) / sqrt(len(Eb / grid_size))))
    print('Variance to energy ratio: {0:.5f}'.format(abs(var(Eb / grid_size) / mean(Eb / grid_size))))
    print('Ising heat capacity: {0:.5f}'.format(Cv))
    print('Ising total magnetization: {0:.5f} +/- {1:0.5f}'.format(mean(Mb / grid_size),
                                                                   std(Mb / grid_size) / sqrt(len(Mb / grid_size))))
    print('Ising susceptibility: {0:.5f}'.format(suscep))

    text1 = 'Expectation value: ' + str(round(mean(Eb / grid_size), 5)) + '+-' + str(
        round(std(Eb / grid_size) / sqrt(len(Eb / grid_size)), 5))
    text2 = 'Expectation value: ' + str(round(mean(Mb / grid_size), 5)) + '+-' + str(
        round(std(Mb / grid_size) / sqrt(len(Mb / grid_size)), 5))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    fig = figure()
    f1 = fig.add_subplot(211)
    f2 = fig.add_subplot(212)
    f1.plot(Eb / grid_size)
    f1.set_xlabel('Block number')
    f1.set_ylabel('Energy per block (au*kb)')
    f1.text(0, -1.3, text1, fontsize=12, bbox=props)

    f2.plot(Mb / grid_size)
    f2.set_xlabel('Block number')
    f2.set_ylabel('Magnetization per block au')
    f2.text(0, 0.3, text2, fontsize=12, bbox=props)
    show()

def problem3(Walkers, Nblocks, Niters, grid_size):
    """
    Calculates the means of the observables and plots them as a function of temperature. Figure in the repo is done
    with 20 points in T linspace. Runtime is manageable (takes a couple of minutes).
    :param Walkers: list of walkers
    :param Nblocks: number of blocks used in the ising model
    :param Niters: number of iterations inside the block
    :param grid_size: size of our grid (lattice)
    :return:
    """

    T = linspace(0.5, 6, 20)  # Temperature (K)

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

def main():
    """
    Main function that runs the program. Problems 2 and 3 are divided into separate functions.
    :return:
    """
    Walkers=[]

    dim = 2
    grid_side = 10
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
    Niters = 1000

    """
    Notice: Energy is measured in units of k_B, which is why
            beta = 1/T instead of 1/(k_B T)
    """

    # Problem 2
    problem2(Walkers, Nblocks, Niters, grid_size)

    # Problem 3
    problem3(Walkers, Nblocks, Niters, grid_size)


if __name__=="__main__":
    main()
        
