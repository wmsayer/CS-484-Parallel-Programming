
#include "basic_defs.h"
#include "databasics.h"

#include "datageneration.h"

#include <stdlib.h>

dist_sort_size_t chooseArraySize(int dist_choice, dist_sort_size_t a, dist_sort_size_t b){

	dist_sort_size_t use_size;

	switch(dist_choice){
		case 0:
			use_size = chooseArraySize_0(a,b);
			break;
		case 1:
			use_size = chooseArraySize_1(a,b);
			break;
		case 2:
			use_size = chooseArraySize_2(a,b);
			break;
		default:
			use_size = chooseArraySize_0(a,b);
	}

	#ifdef SP19
	{//TODO: Remove this next semester, but for this semester, we promised data would be a multiple of the number of ranks.
		// Get number of processes
		int rank, num_ranks;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

		use_size += num_ranks - (use_size % num_ranks);
	}
	#endif

	return use_size;
}

dist_sort_size_t chooseArraySize_0(dist_sort_size_t a, dist_sort_size_t b){

	// Get number of processes
	int rank, num_ranks;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	dist_sort_size_t size;

	size = data_randsize(0, a, b);

	return size;
}
dist_sort_size_t chooseArraySize_1(dist_sort_size_t a, dist_sort_size_t b){
	// Get number of processes
	int rank, num_ranks;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);


	if(0 == rank){//TODO: Randomly choose heavy node so that students don't cheat.
		return b;
	}else{
		return a;
	}
}
dist_sort_size_t chooseArraySize_2(dist_sort_size_t a, dist_sort_size_t b){
	// Get number of processes
	int rank, num_ranks;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
	if(rank % 2 == 0){
		return a;
	}else{
		return b;
	}
}


/**
* A basic data generator. Students may or may not be asked to override this later.
* At test time, we may generate our own data our own way.
*/
void generateData(int dist_choice, dist_sort_t **data, dist_sort_size_t size, dist_sort_size_t min, dist_sort_size_t max){
	//ALLOCATION

	//uint64_t *tmp = (uint64_t*)malloc(size*sizeof(uint64_t));
	dist_sort_t *tmp = (dist_sort_t*)malloc(size*sizeof(dist_sort_t));
	if(tmp == NULL){
		*data = NULL;
		return;
	}else{
		*data = tmp;
	}

	switch(dist_choice){
			case 0:
				generateData_0(*data, size, min, max);
				break;
			case 1:
				generateData_1(*data, size, min, max);
				break;
			case 2:
				generateData_2(*data, size, min, max);
				break;
			case 3:
				generateData_3(*data, size, min, max);
				break;
			default:
				generateData_0(*data, size, min, max);
	}
}



void generateData_0(dist_sort_t *data, dist_sort_size_t size, dist_sort_size_t min, dist_sort_size_t max) {

	//POLUATION
	//Populate data, randomly.
	int i;
	for(i = 0;i< size;i++){
		data[i] = randuint64();//TODO: random generation of the full 64 bits.
	}


}

void generateData_1(dist_sort_t *data, dist_sort_size_t size, dist_sort_size_t min, dist_sort_size_t max) {
	const uint64_t low_numbers_mask = 0x3fffffffffffffff; //Bottom 1/4th of the 64bit uints
	const uint64_t high_start = low_numbers_mask + 500000;
	const uint64_t high_end = high_start + 500000000;
	// Get number of processes
	int rank, nProcs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

	//POLUATION
	//Populate data, randomly.
	if(rank % 2 == 0){
		int i;
		for(i = 0;i< size;i++){
			data[i] = randuint64() & low_numbers_mask;//TODO: random generation of the full 64 bits.
		}
	}else{
		int i;
		for(i = 0;i< size;i++){
			data[i] = randuint64(high_start,high_end);//TODO: random generation of the full 64 bits.
		}
	}
}

void generateData_2(dist_sort_t *data, dist_sort_size_t size, dist_sort_size_t min, dist_sort_size_t max) {
	// Get number of processes
	int rank, nProcs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);


	const dist_sort_t band_size = DIST_SORT_MAX / nProcs;
	dist_sort_t gen_start = rank*band_size;
	dist_sort_t gen_end = gen_start + (band_size/10)*9;

	//POLUATION
	//Populate data, randomly.
	int i;
	for(i = 0;i< size;i++){
		data[i] = randuint64(gen_start, gen_end);//TODO: random generation of the full 64 bits.
	}
}

void generateData_3(dist_sort_t *data, dist_sort_size_t size, dist_sort_size_t min, dist_sort_size_t max) {
	// Get number of processes
	int rank, nProcs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

	const dist_sort_t band_size = DIST_SORT_MAX / nProcs;
	dist_sort_t gen_start = rank*band_size;
	dist_sort_t gen_end = gen_start + (band_size/10)*12;
	if(rank == nProcs -1){
		gen_end = DIST_SORT_MAX;
	}

	//POLUATION
	//Populate data, randomly.
	int i;
	for(i = 0;i< size;i++){
		data[i] = randuint64(gen_start, gen_end);//TODO: random generation of the full 64 bits.
	}
}
