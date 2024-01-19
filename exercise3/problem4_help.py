#! /usr/bin/env python3

from nexus import settings,job,run_project
from nexus import generate_physical_system
from nexus import generate_pwscf
from nexus import generate_pw2qmcpack
from nexus import generate_qmcpack,vmc, dmc
from machine_configs import get_puhti_configs
from optim_params import *

settings(
    pseudo_dir    = './pseudopotentials',
    status_only   = 0,
    generate_only = 1,
    sleep         = 3,
    machine       = '', #!!! should not be empty
    account        = '', #!!! should not be empty
    )

jobs = get_puhti_configs()

# The 4 commented lines below are as before in the vmc diamond case (Problem 3)
#dia16 = generate_physical_system()
#scf = generate_pwscf()
#conv = generate_pw2qmcpack()
#qmc = generate_qmcpack()

# Look into the needed imports and other settings when modifying your workflow scritps

# this loads some optimization parameters from optim_params
optims = getOptims()

# this performs optimization for one and two body Jastrow
optJ2 = generate_qmcpack(
    path = 'diamond/optJ2',
    spin_polarized=False,
    identifier = 'opt',
    job = jobs['optim'], 
    pseudos = ['C.BFD.xml'],
    system = dia16,
    input_type = 'basic',
    twistnum   = 0,
    corrections = [],
    jastrows = [('J1','bspline',6),
                ('J2','bspline',6)],
    calculations = optims,
    dependencies = (conv,'orbitals')
)

# This is the eventual "production/preproduction type" calculation.
# That is, after optimization, you run VMC, and a couple of DMC runs with 
# different time-steps (usually two to four time-steps) in order to look into
# the time-step extrapolation/convergence. 
dmc_run = generate_qmcpack(
    path = 'diamond/dmc',
    spin_polarized=False,
    identifier = 'qmc',
    job = jobs['dmc'], 
    pseudos = ['C.BFD.xml'],
    system = dia16,
    input_type = 'basic',
    estimators = [],
    corrections = [],
    jastrows = [],
    calculations = [
        vmc(
            timestep         = 0.3,
            warmupsteps      = 10,
            blocks           = 80,
            steps            = 5,
            substeps         = 3,
            samples          = 2048,
        ),
        dmc(
            timestep         = 0.01,
            warmupsteps      = 10,
            blocks           = 80,
            steps            = 5,
            nonlocalmoves    = True,
        ),
        dmc(
            timestep         = 0.005,
            warmupsteps      = 50,
            blocks           = 80,
            steps            = 5,
            nonlocalmoves    = True,
        ),
    ],
    dependencies = [(conv,'orbitals'),
                    (optJ2,'jastrow')]
)

#run_project(scf,conv,qmc,optJ2,dmc_run) # either this way or the one below
run_project()
