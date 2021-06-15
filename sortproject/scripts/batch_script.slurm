#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -N proj-benchmark
#PBS -j oe
#PBS -q cs
#PBS -S /projects/cs/cs484/sing_shell.sh
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
#SBATCH --job-name proj-benchmark
#SBATCH -p cs

#TODO: create a nodefile and populate PBS_NUM_NODES/PBS_NUM_PPN if not running in torque.

export TOTAL_CPUS=$(( ${PBS_NUM_NODES:-1} * ${PBS_NUM_PPN:-4} ))

## If not started with PBS, figure out where we are relative to the build directory
##### Snippet from:   http://stackoverflow.com/questions/59895/
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
##### End snippet

#IF SLURM_SUBMIT_DIR is not set, we are not running in SLURM. Die.
if [ "x${SLURM_SUBMIT_DIR}" -eq "x"]; then
	echo "Not running in SLURM. Terminating."
	echo "Not running in SLURM. Terminating." 1>&2
	exit 1
fi

# Moves to the directory the user was in when they ran qsub
cd ${SLURM_SUBMIT_DIR} # Assumed to be the source tree

# Check that the script was submit from the right place.
if [ -d "./cmake" ] && [ -d "./tests" ] && [ -d "./writeup" ]
then
	echo "We seem to be in the right place."
else
	echo "Not submit from the right place! Submit from the root of your repository."
	exit 1
fi

# Creates an out-of-tree build directory for CMake and moves to it
mkdir -p ${SLURM_SUBMIT_DIR}/build
pushd ${SLURM_SUBMIT_DIR}/build

# Bbuild the programs (into the build directory, IE, the current directory)
# then benchmark them. Quit early on failure.
echo "Compiling"
/projects/cs/cs484/sing_exec.sh cmake ${SLURM_SUBMIT_DIR} && /projects/cs/cs484/sing_exec.sh make

# Google test tests
echo "Testing"
srun --mpi=pmi2 --ntasks-per-node ${SLURM_NTASKS_PER_NODE} \
	--cpu-bind=cores --ntasks 4 \
	/projects/cs/cs484/sing_exec.sh ./bin/dotests || ( echo "No use benchmarking an incorrect program." ; exit 1 )

# Benchmarking
# You should change the parameters below to s_mpirun according to how your program mixes MPI and OpenMP.
# The code below assumes 1 MPI rank per CPU core.
echo "Benchmarking"
echo "Benchmark results" >> ${SLURM_SUBMIT_DIR}/writeup/benchmark.txt
srun --mpi=pmi2 --ntasks-per-node 24 \
	--cpu-bind=cores --ntasks 16 \
	/projects/cs/cs484/sing_exec.sh ./bin/sorter -z 2 -a 10000000 -b 20000000 -s 1| grep "Duration" >> ${SLURM_SUBMIT_DIR}/writeup/benchmark.txt