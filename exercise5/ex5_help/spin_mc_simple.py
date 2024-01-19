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


from numpy import *
from matplotlib.pyplot import *

class Walker:
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
    

def Energy(Walkers):
    E = 0.0
    J = 4.0 # given in units of k_B
    # ADD calculation of energy
    #
    #
    #
    #
    return 

def site_Energy(Walkers,Walker):
    E = 0.0
    J = 4.0 # given in units of k_B
    for k in range(len(Walker.nearest_neighbors)):
        j = Walker.nearest_neighbors[k]
        E += -J*Walker.spin*Walkers[j].spin
    return E

def ising(Nblocks,Niters,Walkers,beta):
    M = len(Walkers)
    Eb = zeros((Nblocks,))
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
            #
            #
            #
            #

            Walkers[site].spin = 1.0*s_new

            E_new = site_Energy(Walkers,Walkers[site])

            # ADD Metropolis Monte Carlo
            #
            #
            #
            #
            #
            #
            #
            #

            if j % obs_interval == 0:
                E_tot = Energy(Walkers)/M # energy per spin
                Eb[i] += E_tot
                EbCount += 1
            
        Eb[i] /= EbCount
        Accept[i] /= AccCount[i]
        print('Block {0}/{1}'.format(i+1,Nblocks))
        print('    E   = {0:.5f}'.format(Eb[i]))
        print('    Acc = {0:.5f}'.format(Accept[i]))


    return Walkers, Eb, Accept


def main():
    Walkers=[]

    dim = 2
    grid_side = 10
    grid_size = grid_side**dim
    
    # Ising model nearest neighbors only
    mapping = zeros((grid_side,grid_side),dtype=int) # mapping
    inv_map = [] # inverse mapping
    ii = 0
    for i in range(grid_side):
        for j in range(grid_side):
            mapping[i,j]=ii
            inv_map.append([i,j])
            ii += 1
 

    # ADD comment
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
    eq = 20 # equilibration "time"
    T = 3.0
    beta = 1.0/T
    """
    Notice: Energy is measured in units of k_B, which is why
            beta = 1/T instead of 1/(k_B T)
    """
    Walkers, Eb, Acc = ising(Nblocks,Niters,Walkers,beta)

    plot(Eb)
    Eb = Eb[eq:]
    print('Ising total energy: {0:.5f} +/- {1:0.5f}'.format(mean(Eb), std(Eb)/sqrt(len(Eb))))
    print('Variance to energy ratio: {0:.5f}'.format(abs(var(Eb)/mean(Eb)))) 
    show()

if __name__=="__main__":
    main()
        
