from numpy import *
import h5py

def create_hdf5_file(fname):
    x=linspace(0.0,1.0,101)
    Ne=4
    orbitals = []
    for i in range(Ne):
        orbitals.append(1.0*i*x)

    f = h5py.File(fname,"w")
    gset = f.create_dataset("grid",data=x,dtype='f')
    gset.attrs["info"] = '1D grid'
    
    oset = f.create_dataset("orbitals",shape=(len(x),Ne),dtype='f')
    oset.attrs["info"] = '1D orbitals as (len(grid),N_electrons)'
    for i in range(len(orbitals)):
        oset[:,i]=orbitals[i]
    
    f.close()

def read_hdf5_file(fname):
    f = h5py.File(fname,"r")
    print('Keys in hdf5 file: ',list(f.keys()))
    x = array(f["grid"])
    orbs = array(f["orbitals"])
    orbitals = []
    for i in range(len(orbs[0,:])):
        orbitals.append(orbs[:,i])
    f.close()

def main():
    fname='test.hdf5'
    create_hdf5_file(fname)
    read_hdf5_file(fname)

if __name__=="__main__":
    main()
    
