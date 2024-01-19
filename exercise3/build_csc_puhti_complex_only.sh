#!/bin/bash

################################################################
## * This script builds available configurations of QMCPACK   ##
##   on Puhti at CSC                                          ##
##                                                            ##
## * Execute this script in trunk/                            ##
##   ./config/build_csc_puhti_complex_only.sh                 ##
##                                                            ##
## Last modified: March, 2022                                 ##
################################################################

module load gcc/9.1.0
module load hpcx-mpi/2.4.0
Module load intel-mkl/2019.0.4
module load hdf5/1.10.4-mpi
module load fftw/3.3.8-omp
module load boost/1.68.0-mpi
module load cmake/3.21.3
module load python-env
module load git

CMAKE_FLAGS="-DCMAKE_C_COMPILER=mpicc \ 
             -DCMAKE_CXX_COMPILER=mpicxx"

# Configure and build cpu complex 
echo ""
echo ""
echo "building complex qmcpack for cpu on Puhti"
mkdir -p build_csc_puhti_complex_only
cd build_csc_puhti_complex_only
cmake -DQMC_COMPLEX=1 $CMAKE_FLAGS ..
make -j 12 
cd ..
ln -sf ./build_csc_puhti_complex_only/bin/qmcpack ./qmcpack_csc_puhti_complex
