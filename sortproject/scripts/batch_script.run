#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l nodes=2:ppn=20
#PBS -N proj-benchmark
#PBS -j oe
#PBS -q cs
#PBS -S /projects/cs/cs484/sing_shell.sh

#TODO: create a nodefile and populate PBS_NUM_NODES/PBS_NUM_PPN if not running in torque.

export TOTAL_CPUS=$(( ${PBS_NUM_NODES:-1} * ${PBS_NUM_PPN:-4} ))

## If not started with PBS, figure out where we are relative to the build directory
##### Snippet from:   http://stackoverflow.com/questions/59895/
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
##### End snippet
#IF PBS_O_WORKDIR is not set, we are not running in PBS, choose directory relative to script.
PBS_O_WORKDIR=${PBS_O_WORKDIR:-${SCRIPT_DIR}/..}

# Moves to the directory the user was in when they ran qsub
cd ${PBS_O_WORKDIR} # Assumed to be the source tree

# Check that the script was submit from the right place.
if [ -d "./cmake" ] && [ -d "./tests" ] && [ -d "./writeup" ]
then
	echo "We seem to be in the right place."
else
	echo "Not submit from the right place! Submit from the root of your repository."
	exit 1
fi

# Creates an out-of-tree build directory for CMake and moves to it
mkdir -p ${PBS_O_WORKDIR}/build
pushd ${PBS_O_WORKDIR}/build

# Bbuild the programs (into the build directory, IE, the current directory)
# then benchmark them. Quit early on failure.
echo "Compiling"
cmake ${PBS_O_WORKDIR} && make

# Google test tests
echo "Testing"
mpirun -n 4 ./bin/dotests || ( echo "No use benchmarking an incorrect program." ; exit 1 )

# Benchmarking
# You should change the parameters below to s_mpirun according to how your program mixes MPI and OpenMP.
# The code below assumes 1 MPI rank per CPU core.
echo "Benchmarking"
echo "Benchmark results" >> ${PBS_O_WORKDIR}/writeup/benchmark.txt
s_mpirun -ppn ${PBS_NUM_PPN} -n ${TOTAL_CPUS} ./bin/sorter | grep "Duration" >> ${PBS_O_WORKDIR}/writeup/benchmark.txt
