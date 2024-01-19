"""
Simple VMC and DMC code for Computational Physics 2 course at TAU

- Fill in more comments based on lecture slides
- Follow assignment instructions
-- e.g., the internuclear distance should be 1.4 instead of 1.5
   and currently a=1, but you should use 1.1 at first


By Ilkka Kylanpaa
"""

from numpy import *
from matplotlib import pyplot as plt

#   Walkers represent the different parameters of the system: number of electrons, their positions, spins
#   number of nuclei, their positions and their atomic number.

class Walker:

    #  The constructor
    def __init__(self,*args,**kwargs):
        self.Ne = kwargs['Ne']
        self.Re = kwargs['Re']
        self.spins = kwargs['spins']
        self.Nn = kwargs['Nn']
        self.Rn = kwargs['Rn']
        self.Zn = kwargs['Zn']
        self.sys_dim = kwargs['dim']

    # Copies the attributes of the walker. Used in DMC
    def w_copy(self):
        return Walker(Ne=self.Ne,
                      Re=self.Re.copy(),
                      spins=self.spins.copy(),
                      Nn=self.Nn,
                      Rn=self.Rn.copy(),
                      Zn=self.Zn,
                      dim=self.sys_dim)
    

def vmc_run(Nblocks,Niters,Delta,Walkers_in,Ntarget):
    """
    This function performs the variatonal Monte Carlo run.
    :param Nblocks: Number of blocks
    :param Niters: Number of iterations per block
    :param Delta: The time step
    :param Walkers_in: Walkers given to the run. Only one in VMC
    :param Ntarget: Maximum amount of walkers
    :return: Walkers_out, Eb, Accept
    """

    Eb = zeros((Nblocks,))
    Accept = zeros((Nblocks,))

    vmc_walker = Walkers_in[0] # just one walker needed
    Walkers_out = []
    for i in range(Nblocks):
        for j in range(Niters):
            # moving only electrons
            for k in range(vmc_walker.Ne):
                R = vmc_walker.Re[k].copy()
                Psi_R = wfs(vmc_walker)

                # move the particle
                vmc_walker.Re[k] = R + Delta*(random.rand(vmc_walker.sys_dim)-0.5)
                
                # calculate wave function at the new position
                Psi_Rp = wfs(vmc_walker)

                # calculate the sampling probability
                A_RtoRp = min((Psi_Rp/Psi_R)**2,1.0)
                
                # Metropolis
                if (A_RtoRp > random.rand()):
                    Accept[i] += 1.0/vmc_walker.Ne
                else:
                    vmc_walker.Re[k] = R
                #end if
            #end for
            Eb[i] += E_local(vmc_walker)
        #end for
        if (len(Walkers_out)<Ntarget):
            Walkers_out.append(vmc_walker.w_copy())
        Eb[i] /= Niters
        Accept[i] /= Niters
        print('Block {0}/{1}'.format(i+1,Nblocks))
        print('    E   = {0:.5f}'.format(Eb[i]))
        print('    Acc = {0:.5f}'.format(Accept[i]))
    
    return Walkers_out, Eb, Accept


def dmc_run(Nblocks,Niters,Walkers,tau,E_T,Ntarget):
    """
    This function runs the DIffusion Monte Carlo. Similalry to the VMC it returns the Walkers, energies of the
    different blocks and acceptance ratios.
    :param Nblocks: Number of blocks
    :param Niters: Number of iterations
    :param Walkers: Number of Walkers
    :param tau: Imaginary time
    :param E_T: The mean energy of the blocks calculated by a vmc run.
    :param Ntarget: Maximum amount of walkers
    :return: Walkers_out, Eb, Accept
    """

    max_walkers = 2*Ntarget
    lW = len(Walkers)
    while len(Walkers)<Ntarget:  # Create more walkers.
        Walkers.append(Walkers[max(1,int(lW*random.rand()))].w_copy())

    Eb = zeros((Nblocks,))
    Accept = zeros((Nblocks,))
    AccCount = zeros((Nblocks,))
    
    obs_interval = 5  # observation interval. Averting the correlation
    mass = 1

    for i in range(Nblocks):
        EbCount = 0
        for j in range(Niters):
            Wsize = len(Walkers)
            Idead = []
            for k in range(Wsize):
                Acc = 0.0
                for i_np in range(Walkers[k].Ne):
                    R = Walkers[k].Re[i_np].copy()
                    Psi_R = wfs(Walkers[k])
                    DriftPsi_R = 2*Gradient(Walkers[k],i_np)/Psi_R*tau/2/mass
                    E_L_R = E_local(Walkers[k])

                    DeltaR = random.randn(Walkers[k].sys_dim)
                    logGf = -0.5*dot(DeltaR,DeltaR)

                    # Moving the electron
                    Walkers[k].Re[i_np] = R+DriftPsi_R+DeltaR*sqrt(tau/mass)
                    
                    Psi_Rp = wfs(Walkers[k])
                    DriftPsi_Rp = 2*Gradient(Walkers[k],i_np)/Psi_Rp*tau/2/mass
                    E_L_Rp = E_local(Walkers[k])
                    
                    DeltaR = R-Walkers[k].Re[i_np]-DriftPsi_Rp
                    logGb = -dot(DeltaR,DeltaR)/2/tau*mass

                    A_RtoRp = min(1, (Psi_Rp/Psi_R)**2*exp(logGb-logGf))  # sampling probability.

                    if (A_RtoRp > random.rand()):
                        Acc += 1.0/Walkers[k].Ne
                        Accept[i] += 1
                    else:
                        Walkers[k].Re[i_np] = R
                    
                    AccCount[i] += 1
                
                tau_eff = Acc*tau
                GB = exp(-(0.5*(E_L_R+E_L_Rp) - E_T)*tau_eff)
                MB = int(floor(GB + random.rand()))
                
                if MB>1:
                    for n in range(MB-1):
                        if (len(Walkers) < max_walkers):
                            Walkers.append(Walkers[k].w_copy())
                elif MB==0:
                    Idead.append(k)
 
            Walkers = DeleteWalkers(Walkers,Idead)

            # Calculate observables every now and then
            if j % obs_interval == 0:
                EL = Observable_E(Walkers)
                Eb[i] += EL
                EbCount += 1
                E_T += 0.01/tau*log(Ntarget/len(Walkers))
                

        Nw = len(Walkers)
        dNw = Ntarget-Nw
        for kk in range(abs(dNw)):
            ind = int(floor(len(Walkers)*random.rand()))
            if (dNw>0):
                Walkers.append(Walkers[ind].w_copy())
            elif dNw<0:
                Walkers = DeleteWalkers(Walkers,[ind])

        
        Eb[i] /= EbCount
        Accept[i] /= AccCount[i]
        print('Block {0}/{1}'.format(i+1,Nblocks))
        print('    E   = {0:.5f}'.format(Eb[i]))
        print('    Acc = {0:.5f}'.format(Accept[i]))


    return Walkers, Eb, Accept

def DeleteWalkers(Walkers,Idead):
    if (len(Idead)>0):
        if (len(Walkers)==len(Idead)):
            Walkers = Walkers[0]
        else:
            Idead.sort(reverse=True)   
            for i in range(len(Idead)):
                del Walkers[Idead[i]]

    return Walkers

def H_1s(r1,r2):
    """
    Hydrogen 1s orbital
    :param r1: Walker position 1
    :param r2: Walker position 2
    :return:
    """
    global a
    return exp(-a*sqrt(sum((r1-r2)**2)))
     
def wfs(Walker):
    """
    This function calculates the wave function.
    :param Walker: The walker which wave function we wish to calculate.
    :return: wave function
    """
    # H2 approx
    f = H_1s(Walker.Re[0],Walker.Rn[0])+H_1s(Walker.Re[0],Walker.Rn[1])
    f *= (H_1s(Walker.Re[1],Walker.Rn[0])+H_1s(Walker.Re[1],Walker.Rn[1]))

    J = 0.0 # Jastrow factor. Includes electron correlation to the run.
    # Jastrow e-e
    for i in range(Walker.Ne-1):
        for j in range(i+1,Walker.Ne):
           r = sqrt(sum((Walker.Re[i]-Walker.Re[j])**2))
           if (Walker.spins[i]==Walker.spins[j]):
               J += 0.25*r/(1.0+1.0*r)
           else:
               J += 0.5*r/(1.0+1.0*r)
    
    # Jastrow e-Ion
    for i in range(Walker.Ne):
        for j in range(Walker.Nn):
           r = sqrt(sum((Walker.Re[i]-Walker.Rn[j])**2))
           J -= Walker.Zn[j]*r/(1.0+100*r)
       

    return f*exp(J)

def potential(Walker):
    """
    Calculates the potential energy of the walker
    :param Walker: Given walker
    :return: V (potential energy)
    """
    V = 0.0
    r_cut = 1.0e-4
    # e-e
    for i in range(Walker.Ne-1):
        for j in range(i+1,Walker.Ne):
            r = sqrt(sum((Walker.Re[i]-Walker.Re[j])**2))
            V += 1.0/max(r_cut,r)

    # e-Ion
    for i in range(Walker.Ne):
        for j in range(Walker.Nn):
            r = sqrt(sum((Walker.Re[i]-Walker.Rn[j])**2))
            V -= Walker.Zn[j]/max(r_cut,r)

    # Ion-Ion
    for i in range(Walker.Nn-1):
        for j in range(i+1,Walker.Nn):
            r = sqrt(sum((Walker.Rn[i]-Walker.Rn[j])**2))
            V += 1.0/max(r_cut,r)

    return V

def Local_Kinetic(Walker):
    """
    Calculates the kinetic energy of the walker.
    :param Walker: Given walker
    :return: kinetic energy
    """
    # laplacian -0.5 \nabla^2 \Psi / \Psi
    h = 0.001
    h2 = h*h
    K = 0.0
    Psi_R = wfs(Walker)
    for i in range(Walker.Ne):
        for j in range(Walker.sys_dim):
            Y=Walker.Re[i][j]
            Walker.Re[i][j]-=h
            wfs1 = wfs(Walker)
            Walker.Re[i][j]+=2.0*h
            wfs2 = wfs(Walker)
            K -= 0.5*(wfs1+wfs2-2.0*Psi_R)/h2
            Walker.Re[i][j]=Y
    return K/Psi_R

def Gradient(Walker,particle):
    h=0.001
    dPsi = zeros(shape=shape(Walker.Re[particle]))
    for i in range(Walker.sys_dim):
        Y=Walker.Re[particle][i]
        Walker.Re[particle][i]-=h
        wfs1=wfs(Walker)
        Walker.Re[particle][i]+=2.0*h
        wfs2=wfs(Walker)
        dPsi[i] = (wfs2-wfs1)/2/h
        Walker.Re[particle][i]=Y

    return dPsi

def E_local(Walker):
    """
    Calculates the local energy of the walker
    :param Walker: Given walker
    :return:
    """
    return Local_Kinetic(Walker)+potential(Walker)

def Observable_E(Walkers):
    """ Used in DMC for Walker distribution. """
    E=0.0
    Nw = len(Walkers)
    for i in range(Nw):
        E += E_local(Walkers[i])
    E /= Nw
    return E

def vmc_runs_diff_a(Walkers, Ntarget, vmc_time_step):
    """
    Runs the vmc with different a values, compares them and plots them.

    :param Walkers: Walkers of the system
    :param Ntarget: Maximum amount of walkers
    :param vmc_time_step: time stepped related to Green function and propagator approximation?
    :return:
    """

    a_globals = [1.1, 1.2, 1.3, 1.4, 1.5]
    Eb_means = []
    Eb_errors = []
    var_e_ratio = []
    global a
    for i, a in enumerate(a_globals):
        Walkers, Eb, Acc = vmc_run(100,50,vmc_time_step,Walkers,Ntarget)
        Eb_means.append(mean(Eb))
        Eb_errors.append(std(Eb)/sqrt(len(Eb)))
        var_e_ratio.append(abs(std(Eb) ** 2 / mean(Eb)))
        print('\nMean acceptance ratio: {0:0.3f} +/- {1:0.3f}'.format(mean(Acc), std(Acc) / sqrt(len(Acc))))
        print('\nVMC total energy: {0:0.3f} +/- {1:0.3f}'.format(mean(Eb), std(Eb) / sqrt(len(Eb))))
        print('\nAbsolute value of the variance to energy ratio: {0:0.3f}'.format(abs(std(Eb) ** 2 / mean(Eb))))  # standard deviation is sqrt(variance)

    fig = plt.figure()
    f1 = fig.add_subplot(121)
    f2 = fig.add_subplot(122)

    f1.errorbar(a_globals, Eb_means, Eb_errors, fmt='*', color='red')
    f1.set_xlabel('a')
    f1.set_ylabel('Total energy')
    f1.set_title('Total energies in relation to different a values')

    f2.scatter(a_globals, var_e_ratio)
    f2.set_xlabel('a')
    f2.set_ylabel('Variance to energy ratio')
    f2.set_title('Variance to energy ration in relation to different a values')
    plt.show()

def main():
    Walkers=[]
    Walkers.append(Walker(Ne=2,
                          Re=[array([0.5,0,0]),array([-0.5,0,0])],
                          spins=[0,1],
                          Nn=2,
                          Rn=[array([-0.7,0,0]),array([0.7,0,0])],
                          Zn=[1.0,1.0],
                          dim=3))

    Ntarget=100
    vmc_time_step = 2.7
    vmc_only = True

    # Function that compares the vmc runs in relation to different a
    vmc_runs_diff_a(Walkers, Ntarget, vmc_time_step)

    # Minimum energy and achieved when a = 1.2. (Not really due to the margin of error, but close enough)
    global a
    a = 1.2
    Walkers, Eb, Ac = vmc_run(100, 50, vmc_time_step, Walkers, Ntarget)
    vmc_only = False

    if not vmc_only:
        Walkers, Eb_dmc, Accept_dmc = dmc_run(10,10,Walkers,0.05,mean(Eb),Ntarget)

    print('Total energies: ')
    print('VMC total energy: {0:0.3f} +/- {1:0.3f}'.format(mean(Eb), std(Eb) / sqrt(len(Eb))))
    print('DMC total energy: {0:0.3f} +/- {1:0.3f}'.format(mean(Eb_dmc), std(Eb_dmc)/sqrt(len(Eb_dmc))))

    # DMC total energy is slightly smaller than VMC total energy and takes a lot longer to calculate due to multiple
    # Walkers of the system.

    # The Jastrow factor that is used is given in the Pade form. Changing the values as instructed to a = 0.5 and
    # b = 0.5 makes the energies smaller and increases the difference between them.

    # The wave function parameters could be improved with more runs. For example increasing the amount of blocks or
    # iterations within a block. This does require more computing power. The laplacian is also numerically approximated,
    # which alters the results. If analytical solutions are achievable they should net more accurate results.

if __name__=="__main__":
    main()
        
