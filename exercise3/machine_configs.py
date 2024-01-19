#! /usr/bin/env python3

# NOTICE: YOU NEED TO ADD THE CORRECT MODULES TO qmc_presub BELOW


from nexus import job

def general_configs(machine):
    if machine=='puhti':
        jobs = get_puhti_configs()
    else:
        print('Using Puhti as defaul machine')
        jobs = get_puhti_configs()
    return jobs

def get_puhti_configs():
    # remember to load the modules used in compiling the code
    # since those are needed in running the code
    scf_presub = '''
module load intel/19.0.4
module load hpcx-mpi/2.4.0
module load intel-mkl/2019.0.4
module load StdEnv
module load hdf5
module load cmake/3.18.2
module load git
    '''

    # what modules are needed for qmcpack
    qmc_presub = '''
module load gcc/9.1.0
module load hpcx-mpi/2.4.0
Module load intel-mkl/2019.0.4
module load hdf5/1.10.4-mpi
module load fftw/3.3.8-omp
module load boost/1.68.0-mpi
module load cmake/3.21.3
module load python-env
module load git
'''

    # application that performs the calculations
    qe_app='pw.x'
    conv_app='pw2qmcpack.x'
    qmc_app='qmcpack_complex'

    # csc queue
    # https://docs.csc.fi/computing/running/batch-job-partitions/
    csc_queue = 'small' # test, small, large, ...

    # define jobs
    # 4 processes for scf, 1 for conv, 20 for vmc, 20 for optim, 20 for dmc
    scf  = job(cores=4,minutes=10,user_env=False,presub=scf_presub,app=qe_app,queue=csc_queue)
    conv = job(cores=1,minutes=10,user_env=False,presub=scf_presub,app=conv_app,queue=csc_queue)
    vmc  = job(cores=20,minutes=10,user_env=False,presub=qmc_presub,app=qmc_app,queue=csc_queue)
    optim  = job(cores=20,minutes=10,user_env=False,presub=qmc_presub,app=qmc_app,queue=csc_queue)
    dmc  = job(cores=20,minutes=20,user_env=False,presub=qmc_presub,app=qmc_app,queue=csc_queue)

    # 40 processes (1 node = 40 processors at Puhti)
    #scf  = job(nodes=1,hours=1,user_env=False,presub=scf_presub,app=qe,queue=csc_queue)
    
    jobs = {'scf' : scf, 'conv': conv, 'vmc': vmc, 'optim': optim, 'dmc': dmc}

    return jobs
