#!/bin/bash

# Install script for QE7.0 on Puhti
# - run this in the qe-7.0 directory
# - will create executables into qe-7.0/build_mpi/bin 

module load intel/19.0.4
module load hpcx-mpi/2.4.0 
module load intel-mkl/2019.0.4 
module load StdEnv
module load hdf5
module load cmake/3.18.2
module load git

GCC_CONF="-gnu-prefix=/appl/spack/install-tree/gcc-4.8.5/gcc-7.4.0-l7gbl5/bin/ -Xlinker -rpath=/appl/spack/install-tree/gcc-4.8.5/gcc-7.4.0-l7gbl5/lib64"

GCC_CFG="$(pwd)/gcc_puhti_conf.cfg"
echo $GCC_CONF > $GCC_CFG

export ICCCFG=$GCC_CFG
export ICPCCFG=$GCC_CFG
export IFORTCFG=$GCC_CFG

mkdir -p build_mpi
cd build_mpi
cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_Fortran_COMPILER=mpif90 ..
make -j 8
