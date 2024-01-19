from nexus import loop, linear

def getOptims():
    linopt1 = linear(
        energy               = 0.95,
        unreweightedvariance = 0.0,
        reweightedvariance   = 0.05,
        timestep             = 0.3,
        samples              = 12800,
        walkers              = 1,
        warmupsteps          = 10,
        blocks               = 100,
        steps                = 1,
        substeps             = 3,
        #gpu                  = True,
        # Disable nonlocalpp for GPU, i.e., nonlocalpp = False,
        nonlocalpp           = True,
        maxweight            = 1e9,
        minmethod            = 'OneShiftOnly',
        minwalkers	     = 0.4,
        usebuffer            = True,
        exp0                 = -2,
        bigchange            = 10.0,
        alloweddifference    = 6e-04,
        stepsize             = 0.05,
        nstabilizers         = 3,
        )
    optims=[]
    optims.append(loop(max=10,qmc=linopt1))

    return optims
