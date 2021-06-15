#include <iostream>
#include <cstdint>
#include <time.h>
#include <random>
#include <chrono>

#include <cassert>

#include <stdio.h>
#include <unistd.h>

#include <mpi.h>

#include "orderedio.h"

#include "basic_defs.h"
#include "databasics.h"
#include "datageneration.h"
#include "solution.h"

#define MPI_PRINT(content) if (rank == 0) std::cout << content
#define MPI_ERROR(content) if (rank == 0) std::cerr << content

int main(int argc, char **argv) {
	// Initialize the MPI environment
	// We are promising that only the main thread of each instance of the program will communicate.
	int provided;
	MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);

	// Get number of processes
	int nProcs;
	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

	// Get rank of the process
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int cmdln_size_dist_choice = 0;
	dist_sort_size_t size_dist_a = 1500000;
	dist_sort_size_t size_dist_b = 3000000;
	int cmdln_dist_choice = 0;
	int cmdln_seed = 0;
	int cmdln_verbosity = 0;
	bool use_cmdline_seed = false;

	int opt;
	// snippet from https://www.geeksforgeeks.org/getopt-function-in-c-to-parse-command-line-arguments/
	while((opt = getopt(argc, argv, ":hd:s:v:z:a:b:")) != -1)
	{
		switch(opt)
		{
			case 'h':
				printf("Usage: %s [-d DISTRIBUTION_INT ] [-v VERBOSITY_INT ] [-s SEED_INT ] [-z SIZE_DISTRIBUTION_INT ] [-a SIZE_A_PARAMETER ] [-b SIZE_B_PARAMETER ]\n", argv[0]);
				exit(0);
				break;
			case 'd':
				cmdln_dist_choice = atoi(optarg);
				if(0 == rank){
					printf("Using distribution choice %i\n",cmdln_dist_choice);
				}
				break;
			case 'z':
				cmdln_size_dist_choice = atoi(optarg);
				if(0 == rank){
					printf("Using size distribution choice %i\n",cmdln_size_dist_choice);
				}
				break;
			case 'a':
				size_dist_a = atol(optarg);
				break;
			case 'b':
				size_dist_b = atol(optarg);
				break;
			case 's':
				cmdln_seed = atoi(optarg);
				use_cmdline_seed = true;
				break;
			case 'v':
				cmdln_verbosity = atoi(optarg);
				if(0>cmdln_verbosity){
					printf("Verbosity (-v) must be >= 0 \n");
					exit(1);
				}
				break;
			case ':':
				printf("option needs a value\n");
				break;
			case '?':
				printf("unknown option: %c\n", optopt);
				break;
		}
	}

	for(; optind < argc; optind++){
		printf("extra arguments: %s\n", argv[optind]);
	}

	assert(size_dist_a <= size_dist_b);




	std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time;

	std::chrono::duration<double> rebalance_time;
	std::chrono::duration<double> sort_time_1;
	std::chrono::duration<double> sort_time_2;
	std::chrono::duration<double> splitters_time;
	std::chrono::duration<double> movedata_time;

	// Initialize the basic C random number generator. Must be different in each process.
	if(use_cmdline_seed){
		srand(cmdln_seed+rank);
	}else{
		unsigned int use_base_seed = time(0);
		srand(use_base_seed+rank);
		if(0==rank){
			std::cout << "Using seed : " << use_base_seed << std::endl;
		}
	}

	// Each rank computes different array sizes
	dist_sort_size_t local_N;
	dist_sort_size_t global_N;

	// The highest splitter will be the maximum value
	dist_sort_t *splitters = (dist_sort_t*)malloc(nProcs*sizeof(dist_sort_t));
	dist_sort_t *counts = (dist_sort_t*)malloc(nProcs*sizeof(dist_sort_t));

	dist_sort_t *starting_data = NULL;
	dist_sort_t *rebalanced_data = NULL;
	dist_sort_size_t rebalanced_data_size; // Should be equal to N / nProcs
	dist_sort_t *partitioned_data = NULL;
	dist_sort_size_t partitioned_data_size;

	uint64_t confirm_mask = randuint64();
	uint64_t confirm_modulus = randuint64() % 0xffffffff;

	// Each rank generates local_N elements with values ranging from 0 to DIST_SORT_MAX
	local_N = chooseArraySize(cmdln_size_dist_choice,size_dist_a,size_dist_b);
	generateData(cmdln_dist_choice,
		&starting_data, local_N, 0, DIST_SORT_MAX);

	// Perform MPI all reduce to sum up all local_N's and get global_N
	MPI_Allreduce(&local_N, &global_N, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
	//std::cout << "Rank " << rank << " has " << local_N << " elements" << std::endl;
	if(cmdln_verbosity >= 1){
		mpi_ordered_printf(MPI_COMM_WORLD, "Rank %i has %llu elements\n",rank, local_N);
	}
	if(0 == rank){
		std::cout << "In total, there are " << global_N << " elements" << std::endl;
	}

	uint64_t gen_checksum = distributed_checksum(starting_data, local_N, confirm_mask, confirm_modulus, MPI_COMM_WORLD);

	// Rebalance data acrosses processes by redistributing
	MPI_PRINT("\nRebalancing...");
	start_time = std::chrono::high_resolution_clock::now();
	MPI_Barrier(MPI_COMM_WORLD);
	rebalance(starting_data, local_N, &rebalanced_data, &rebalanced_data_size);
	MPI_Barrier(MPI_COMM_WORLD);
	end_time = std::chrono::high_resolution_clock::now();
	rebalance_time = end_time - start_time;
	MPI_PRINT("Done.\n");

	uint64_t rebalance_checksum = distributed_checksum(rebalanced_data, rebalanced_data_size, confirm_mask, confirm_modulus, MPI_COMM_WORLD);

	if (gen_checksum != rebalance_checksum) {
		MPI_ERROR("FATAL ERROR: Checksums did not match after rebalancing data. Data must have changed.");
		exit(1);
	} else {
		MPI_PRINT("\t( Checksum matched after rebalancing. )\n");
	}
	if(cmdln_verbosity >= 1){
		mpi_ordered_printf(MPI_COMM_WORLD, "Rank %i has %llu elements after rebalance\n",rank, rebalanced_data_size);
	}

	// Sort local data if necesssary
	MPI_PRINT("\nSorting Rebalanced Data...");
	start_time = std::chrono::high_resolution_clock::now();
	sort(rebalanced_data, rebalanced_data_size);
	end_time = std::chrono::high_resolution_clock::now();
	sort_time_1 = end_time - start_time;
	MPI_PRINT("Done.\n");

	uint64_t sorted_checksum = distributed_checksum(rebalanced_data, rebalanced_data_size, confirm_mask, confirm_modulus, MPI_COMM_WORLD);

	if (gen_checksum != sorted_checksum) {
		MPI_ERROR("FATAL ERROR: Checksums did not match after sorting data. Data must have changed.");
		exit(1);
	} else {
		MPI_PRINT("\t( Checksum matched after sorting. )\n");
	}

	// Apply the (repeated) histogramming algorithm.
	MPI_PRINT("\nHistogramming...");
	start_time = std::chrono::high_resolution_clock::now();
	MPI_Barrier(MPI_COMM_WORLD);
	findSplitters(rebalanced_data, rebalanced_data_size, splitters, counts, nProcs);
	MPI_Barrier(MPI_COMM_WORLD);
	end_time = std::chrono::high_resolution_clock::now();
	splitters_time = end_time - start_time;
	MPI_PRINT("Done.\n");


	//DEBUGGING Splitter and Counts printer
	if(cmdln_verbosity >= 2){
		//output splitter and counts
		for(int i = 0;i<nProcs;++i){
		MPI_Barrier(MPI_COMM_WORLD);
		if(i == rank){
			printf("Splitters: [<0>");
			for(int i = 0;i<nProcs;++i){
				printf(" , %llu",splitters[i]);
			}
			printf(" ]\n");
			printf("Counts: [-");
			for(int i = 0;i<nProcs;++i){
				printf(" , %llu",counts[i]);
			}
			printf(" ]\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);
		}
	}

	//ASSERT that splitters increase monotonically
	if(0 == rank){
		for(int i = 1;i<nProcs;++i){
			if(! (splitters[i-1] <= splitters[i])){
				MPI_ERROR("FATAL ERROR: Splitters are not monotonically increasing.");
				exit(1);
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);



	// Send data to dest proc and receive data in recv_data
	MPI_PRINT("Moving data to target processors...");
	start_time = std::chrono::high_resolution_clock::now();
	MPI_Barrier(MPI_COMM_WORLD);
	moveData(rebalanced_data, rebalanced_data_size, &partitioned_data, &partitioned_data_size, splitters, counts, nProcs);
	MPI_Barrier(MPI_COMM_WORLD);
	end_time = std::chrono::high_resolution_clock::now();
	movedata_time = end_time - start_time;
	MPI_PRINT("Done.\n");

	// Re-sort local data if necesssary
	MPI_PRINT("\nSorting Bucket Data...");
	start_time = std::chrono::high_resolution_clock::now();
	sort(partitioned_data, partitioned_data_size);
	end_time = std::chrono::high_resolution_clock::now();
	sort_time_2 = end_time - start_time;
	MPI_PRINT("Done.\n");

	// Final confirmation
	uint64_t moved_checksum = distributed_checksum(partitioned_data, partitioned_data_size, confirm_mask, confirm_modulus, MPI_COMM_WORLD);
	if (gen_checksum != moved_checksum) {
		MPI_ERROR("FATAL ERROR: Checksums did not match after moving data. Data must have changed.\n");
		exit(1);
	} else {
		MPI_PRINT("\t( Checksum matched after histogramming and sorting. )\n");
	}

	free(starting_data);
	free(splitters);
	free(rebalanced_data);
	free(partitioned_data);

	end_time = std::chrono::high_resolution_clock::now();
	if (rank == 0) {
		std::chrono::duration<double> total_time(0.0);
		std::cout << std::endl;
		std::cout << "Duration_Rebalance " << (rebalance_time).count() << " s" << std::endl << std::flush; total_time+=rebalance_time;
		#ifdef MEASURE_SORT_TIME
		std::cout << "Duration_Sort_1 " << (sort_time_1).count() << " s" << std::endl << std::flush; total_time+=sort_time_1;
		#endif
		std::cout << "Duration_Splitters " << (splitters_time).count() << " s" << std::endl << std::flush; total_time+=splitters_time;
		std::cout << "Duration_Move " << ( movedata_time).count() << " s" << std::endl << std::flush; total_time+=movedata_time;
		#ifdef MEASURE_SORT_TIME
		std::cout << "Duration_Sort_2 " << (sort_time_2).count() << " s" << std::endl << std::flush; total_time+=sort_time_2;
		#endif
		std::cout << "Duration_Total " << (total_time).count() << " s" << std::endl << std::flush;
		//TODO: Remove before next semester. Kept here in case anyone has alredy built scripts that use it.
		std::cout << "Duration " << (total_time).count() << " s" << std::endl << std::flush;
	}

	// Finalize MPI environment.
	MPI_Finalize();
}
