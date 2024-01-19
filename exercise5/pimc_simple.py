""" This program simulates two hydrogen atoms in 3D interacting via Morse potential energy surface. """
import numpy as np
from numpy import *
from matplotlib.pyplot import *
from scipy.special import erf

class Walker:
    """ Create the walkers for the simulation. Form the path integral by connecting to each other.
    Analogous to week 3 VMC and DMC. Now the electrons are treated as protons. """
    def __init__(self,*args,**kwargs):
        self.Ne = kwargs['Ne']
        self.Re = kwargs['Re']
        self.spins = kwargs['spins']
        self.Nn = kwargs['Nn']
        self.Rn = kwargs['Rn']
        self.Zn = kwargs['Zn']
        self.tau = kwargs['tau']
        self.sys_dim = kwargs['dim']

    def w_copy(self):
        return Walker(Ne=self.Ne,
                      Re=self.Re.copy(),
                      spins=self.spins.copy(),
                      Nn=self.Nn,
                      Rn=self.Rn.copy(),
                      Zn=self.Zn,
                      tau=self.tau,
                      dim=self.sys_dim)
    

def kinetic_action(r1,r2,tau,lambda1):
    """
    Short time thermal density matrix is often expressed in terms of so-called action. This is the kinetic part of it.
    :param r1: configuration/coordinate 1
    :param r2: configuration/coordinate 2
    :param tau: the time step
    :param lambda1: constant
    :return: the kinetic action
    """
    return sum((r1-r2)**2)/lambda1/tau/4

def potential_action(Walkers,time_slice1,time_slice2,tau):
    """
    Short time thermal density matrix is often expressed in terms of so-called action. This is the potential part of it.
    :param Walkers:
    :param time_slice1:
    :param time_slice2:
    :param tau: time step
    :return:
    """
    return 0.5*tau*(potential(Walkers[time_slice1]) \
                    +potential(Walkers[time_slice2]))

def move_sampling(r0, r1, r2, sigma2, sys_dim):
    """
    Sampling part of the code turned into a function.
    :param r0: position of the porticle (electron)
    :param r1: position of the porticle (electron)
    :param r2: position of the porticle (electron)
    :param sigma2: variance
    :param sys_dim: dimensions of the system
    :return:
    """
    sigma = sqrt(sigma2)
    r02_ave = (r0 + r2) / 2
    log_S_Rp_R = -sum((r1 - r02_ave) ** 2) / 2 / sigma2
    Rp = r02_ave + random.randn(sys_dim) * sigma
    log_S_R_Rp = -sum((Rp - r02_ave) ** 2) / 2 / sigma2
    return log_S_Rp_R, log_S_R_Rp, Rp

def tuniform_sampling(r1, sigma2, sys_dim):
    """
    Sampling done using uniform function. If this function is used instead of move_sampling the result is not correct.
    So sampling does have an effect in the simulation.
    :param r1: position of the particle (electron)
    :param sigma2: variance
    :param sys_dim: dimensions of the system.
    :return:
    """
    sigma = sqrt(sigma2)
    log_S_Rp_R = 1
    Rp = r1 + (random.rand(sys_dim)-0.5)*sigma
    log_S_R_Rp = 1
    return log_S_Rp_R, log_S_R_Rp, Rp

def pimc(Nblocks,Niters,Walkers):
    """
    Performws the path integral monte carlo.
    :param Nblocks: Number of blocks in the simulation
    :param Niters: Numbers of iterations within a block
    :param Walkers: The walker objects as a list.
    :return:
    """
    M = len(Walkers)  # Trotter number is the same as the number of walkers, in this simulation,
    Ne = Walkers[0].Ne*1
    sys_dim = 1*Walkers[0].sys_dim
    tau = 1.0*Walkers[0].tau  # The time step
    lambda1 = 1/(2*1836)  # Constant hbar^2/2m
    Eb = zeros((Nblocks,))
    rb = zeros((Nblocks,))
    Accept=zeros((Nblocks,))
    AccCount=zeros((Nblocks,))
    sigma2 = lambda1*tau  # variance, related to sampling algorithm in the bisection method. (Lecture notes)
    sigma = sqrt(sigma2)  # standard deviation

    obs_interval = 5  # Related to calculating the observables at the end of this function
    for i in range(Nblocks):
        EbCount = 0
        for j in range(Niters):
            # Picking the "points" in the path integral involved in the move. Path integral intervals are time_slices
            time_slice0 = int(random.rand()*M)
            time_slice1 = (time_slice0+1)%M
            time_slice2 = (time_slice1+1)%M

            # Choose the particle (electron) involved in the move.
            ptcl_index = int(random.rand()*Ne)
            r0 = Walkers[time_slice0].Re[ptcl_index]
            r1 = 1.0*Walkers[time_slice1].Re[ptcl_index]
            r2 = Walkers[time_slice2].Re[ptcl_index]

            # Calculate old position kinetic and potential actions.
            KineticActionOld = kinetic_action(r0,r1,tau,lambda1) +\
                kinetic_action(r1,r2,tau,lambda1)
            PotentialActionOld = potential_action(Walkers,time_slice0,time_slice1,tau)+potential_action(Walkers,time_slice1,time_slice2,tau)

            # bisection sampling / moves. Moves the electron from r1 to r1' and calculates the natural logarithm of
            # transition probabilities. Using the variance and standard deviation defined earlier.
            log_S_Rp_R, log_S_R_Rp, Rp = move_sampling(r0, r1, r2, sigma2, sys_dim)
            #log_S_Rp_R, log_S_R_Rp, Rp = tuniform_sampling(r1, sigma2, sys_dim)  # Uniform sampling.

            # Updates the particle position.
            Walkers[time_slice1].Re[ptcl_index] = 1.0*Rp
            r0 = Walkers[time_slice0].Re[ptcl_index]
            r2 = Walkers[time_slice2].Re[ptcl_index]
            # Calculates the new actions for new positions.
            KineticActionNew = kinetic_action(r0,Rp,tau,lambda1) +\
                kinetic_action(Rp,r2,tau,lambda1)
            PotentialActionNew = potential_action(Walkers,time_slice0,time_slice1,tau)+potential_action(Walkers,time_slice1,time_slice2,tau)

            # Calculate the differences between old and new actions.
            deltaK = KineticActionNew-KineticActionOld
            deltaU = PotentialActionNew-PotentialActionOld

            # Metropolis MC. Analogous to last week VMC and PMC.
            q_R_Rp = exp(log_S_Rp_R-log_S_R_Rp-deltaK-deltaU)
            A_RtoRp = min(1.0,q_R_Rp)
            if (A_RtoRp > random.rand()):
                Accept[i] += 1.0
            else:
                Walkers[time_slice1].Re[ptcl_index]=1.0*r1
            AccCount[i] += 1

            # calculate observables.
            if j % obs_interval == 0:
                r_distance = calculate_distance(r0, Walkers[time_slice1].Re[ptcl_index], r2)
                E_kin, E_pot = Energy(Walkers)
                #print(E_kin,E_pot)
                Eb[i] += E_kin + E_pot
                rb[i] += r_distance
                EbCount += 1
            #exit()

        # Block averages
        Eb[i] /= EbCount
        rb[i] /= EbCount
        Accept[i] /= AccCount[i]
        print('Block {0}/{1}'.format(i+1,Nblocks))
        print('    E   = {0:.5f}'.format(Eb[i]))
        print('    Acc = {0:.5f}'.format(Accept[i]))


    return Walkers, Eb, rb, Accept


def calculate_distance(r0, r1, r2):
    """
    Calculates the average value of the internuclear distance after Metropolis MC.
    :param r0: position of time slice 0
    :param r1: position of time slice 1
    :param r2: position of time slice 2
    :return:
    """
    r0_to_r1 = linalg.norm(r0-r1)
    r0_to_r2 = linalg.norm(r0-r2)
    r1_to_r2 = linalg.norm(r1-r2)
    r_avg = (r0_to_r1 + r0_to_r2 + r1_to_r2)/3
    return r_avg


def Energy(Walkers):
    """
    Calculates the energy averages related to the walkers
    :param Walkers: Walkers as a list
    :return: kinetic energy and potential energy averages of all the walkers.
    """
    M = len(Walkers)
    d = 1.0*Walkers[0].sys_dim
    tau = Walkers[0].tau
    lambda1 = 1/(2*1836)
    U = 0.0
    K = 0.0
    for i in range(M):
        U += potential(Walkers[i])
        for j in range(Walkers[i].Ne):
            if (i<M-1):
                K += d/2/tau-sum((Walkers[i].Re[j]-Walkers[i+1].Re[j])**2)/4/lambda1/tau**2
            else:
                K += d/2/tau-sum((Walkers[i].Re[j]-Walkers[0].Re[j])**2)/4/lambda1/tau**2    
    return K/M,U/M
        
def morse_potential(r):
    """
    Calculates the interaction via morse potential energy surface
    :param r: distance between atoms
    :return:
    """
    De = 0.1745
    re = 1.4
    a = 1.0282
    Um = De*(1-np.exp(-a*(r-re)))**2 - De
    return Um

def potential(Walker):
    """
    Calculates the potential of a single walker.
    :param Walker: list of walkers
    :return: potential
    """
    V = 0.0
    r_cut = 1.0e-12
    # e-e
    for i in range(Walker.Ne-1):
        for j in range(i+1,Walker.Ne):
            r = sqrt(sum((Walker.Re[i]-Walker.Re[j])**2))
            V += morse_potential(r)

    return V

def external_potential(Walker):
    """
    Calculates the external potential of a single walker
    :param Walker: list of walkers
    :return: external potential
    """
    V = 0.0
    for i in range(Walker.Ne):
        V += 0.5*sum(Walker.Re[i]**2)
        
    return V

def run(M, time_step):
    """
    Runs the path integral monte carlo.
    :return:
    """
    Walkers=[]

    # For 3D quantum dot
    Walkers.append(Walker(Ne=2,
                        Re=[array([0.5,0, 0]),array([-0.5,0, 0])],
                        spins=[0,1],
                        Nn=2, # not used
                        Rn=[array([-0.7,0]),array([0.7,0])], # not used
                        Zn=[1.0,1.0], # not used
                        tau = time_step,
                        dim=3))

    for i in range(M-1):
         Walkers.append(Walkers[i].w_copy())
    Nblocks = 200
    Niters = 100
    
    Walkers, Eb, rb, Acc = pimc(Nblocks,Niters,Walkers)

    fig = figure()
    f1 = fig.add_subplot(211)
    f2 = fig.add_subplot(212)
    f1.plot(Eb)
    conv_cut=50
    f1.axvline(x=conv_cut, color='k', linestyle='--')
    #f1.plot([conv_cut,conv_cut],gca().get_ylim(),'k--')
    f1.set_xlabel('Block number')
    f1.set_ylabel('Energy (au)')
    M_str = str(M)
    f1.set_title('M = ' + M_str)

    f2.plot(rb, color='r')
    f2.set_xlabel('Block number')
    f2.set_ylabel('Internuclear distance (au)')

    Eb = Eb[conv_cut:]
    print('PIMC total energy: {0:.5f} +/- {1:0.5f}'.format(mean(Eb), std(Eb)/sqrt(len(Eb))))
    print('Variance to energy ratio: {0:.5f}'.format(abs(var(Eb)/mean(Eb)))) 
    show()

def main():

    # Comparison of energetics and internuclear distance for classical and quantum atoms.
    # Constant T = 300K, M = [1, 8, 16]

    kb = 3.16681 * 10 ** (-6)
    T = 300
    M = [1, 8, 16]
    for i in range(len(M)):
        time_step = 1/(kb*M[i]*T)
        run(M[i], time_step)

if __name__=="__main__":
    main()
        
