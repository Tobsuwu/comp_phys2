#! /usr/bin/env python3

from nexus import settings,run_project,job
from nexus import generate_physical_system
from nexus import generate_pwscf
from machine_configs import get_puhti_configs

settings(
    pseudo_dir    = './pseudopotentials',
    results       = '',
    status_only   = 0,
    generate_only = 1,
    sleep         = 3,
    machine       = '',
    account       = ''
    )

jobs = get_puhti_configs()

dia16 = generate_physical_system(
    units  = 'A',
    axes   = [[ 1.785,  1.785,  0.   ],
              [ 0.   ,  1.785,  1.785],
              [ 1.785,  0.   ,  1.785]],
    elem   = ['C','C'],
    pos    = [[ 0.    ,  0.    ,  0.    ],
              [ 0.8925,  0.8925,  0.8925]],
    tiling = (2,2,2),
    kgrid  = (1,1,1),
    kshift = (0,0,0),
    C      = 4
    )
              
scf = generate_pwscf(
    identifier   = 'scf',
    path         = 'scf',
    job          = jobs['scf'],
    input_type   = 'generic',
    calculation  = 'scf',
    input_dft    = 'lda', 
    ecutwfc      = 200,   
    conv_thr     = 1e-8, 
    nosym        = True,
    wf_collect   = True,
    system       = dia16,
    kgrid        = (1,1,1),
    pseudos      = ['C.BFD.upf'], 
    )

run_project(scf)
