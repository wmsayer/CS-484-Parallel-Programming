#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <stdint.h>

#include <array>
#include <random>
#include <chrono>

#include "databasics.h"

/**
* A triangle distribution with mean mean;
* https://en.wikipedia.org/wiki/Triangular_distribution
*/
uint64_t data_randsize(double min, double max, double mean){
	//adapted from http://www.cplusplus.com/reference/random/piecewise_linear_distribution/

	double mode_from_mean = (mean*3.0) - min - max;
	//printf("the mode is %f , %f, %f, %f;\n",mode_from_mean, min, max, mean);

	// obtain a seed from the system clock:
	unsigned seed1 = rand(); //+std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 g1 (seed1);

	//std::default_random_engine generator;
	std::array<double,3> intervals {(double)min, (double)mode_from_mean, (double)max};
	std::array<double,3> weights {0.0, 1.0, 0.0};
	std::piecewise_linear_distribution<double>distribution(intervals.begin(),intervals.end(),weights.begin());

	//return (uint32_t)distribution(generator);
	return (uint32_t)distribution(g1);
}

/**
*Returns a random uint64_t where every bit is a random bit.
*
*/
uint64_t randuint64(){
	const uint64_t rMASK = 0x7fff;
	return ( ((uint64_t)rand()&rMASK ) ) |
			( ((uint64_t)rand()&rMASK) << 15 ) |
			( ((uint64_t)rand()&rMASK) << 30 ) |
			( ((uint64_t)rand()&rMASK) << 45 ) |
			( ((uint64_t)rand()&rMASK) << 60 );
}

uint64_t randuint64(uint64_t min, uint64_t max){
	return randuint64()%(max-min) + min;
}

void allocate_and_populate(uint64_t **data, const uint64_t size){
	int i;

	uint64_t *tmp = (uint64_t*)malloc(size*sizeof(uint64_t));
	if(tmp == NULL){
		*data = NULL;
		return;
	}

	*data = tmp;

	//Populate data, randomly.
	for(i = 0;i< size;i++){
		tmp[i] = randuint64();//TODO: random generation of the full 64 bits.
	}


}

uint64_t local_checksum(const uint64_t *data,
	const uint64_t my_size,
	const uint64_t mask, const uint64_t modulus){
		size_t i;
		uint64_t result = 0;

		//printf("DCHKSUM I have mask : %llu and modulus %llu size %llu\n",mask, modulus,my_size);

		//local result
		for(i = 0;i< my_size;i++){
			result = (result + (mask ^ data[i])%modulus ) % modulus;
		}
	return result;
}

uint64_t distributed_checksum(const uint64_t *data,
	const uint64_t my_size,
	const uint64_t mask, const uint64_t modulus,
	MPI_Comm comm)
{
	uint64_t use_mask = mask;
	uint64_t use_modulus = modulus & (uint64_t)0x3fffffffffffffff;
	//distribute the leader's mask and modulus to all nodes.
	MPI_Bcast(&use_mask,1,MPI_UNSIGNED_LONG_LONG,0,comm);
	MPI_Bcast(&use_modulus,1,MPI_UNSIGNED_LONG_LONG,0,comm);

	//printf("DCHKSUM I have mask : %llu and modulus %llu \n",use_mask, use_modulus);

	uint64_t result = local_checksum(data,my_size,use_mask,use_modulus);

	// Get the rank of the process
	int comm_rank, comm_size;

	MPI_Comm_rank(comm, &comm_rank);
	MPI_Comm_size(comm, &comm_size);

	uint64_t *tmp;

	//We could figure out how to use a proper MPI reduction later.
	if( 0 == comm_rank ){
		tmp = (uint64_t*)malloc(sizeof(uint64_t)*comm_size);
	}

	MPI_Gather(&result,1,MPI_UNSIGNED_LONG_LONG,
				tmp,1,MPI_UNSIGNED_LONG_LONG,
				0,comm);

	if( 0 == comm_rank ){
		result = local_checksum(tmp,comm_size,0,use_modulus);
		free(tmp);
	}
	MPI_Bcast(&result,1,MPI_UNSIGNED_LONG_LONG,0,comm);

	return result;
}
