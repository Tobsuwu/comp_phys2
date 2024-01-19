#! /usr/bin/env python3
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
    # these are needed in running the code
    scf_presub = '''
module load intel/19.0.4
module load hpcx-mpi/2.4.0
module load intel-mkl/2019.0.4
module load StdEnv
module load hdf5
module load cmake/3.18.2
module load git
    '''

    # application that performs the calculations
    qe='pw.x'

    # csc queue
    # https://docs.csc.fi/computing/running/batch-job-partitions/
    csc_queue = 'small' # test, small, large, ...

    # define job
    # for this 4 processes is enough
    scf  = job(cores=4,minutes=5,user_env=False,presub=scf_presub,app=qe,queue=csc_queue)

    # As an example, 40 processes (1 node = 40 processors at Puhti)
    #scf  = job(nodes=1,hours=1,user_env=False,presub=scf_presub,app=qe,queue=csc_queue)
    
    jobs = {'scf' : scf}

    return jobs
