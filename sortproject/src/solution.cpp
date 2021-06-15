#include <cmath>
#include <algorithm>
#include <iostream>
#include <utility>
#include <queue>
#include <vector>

#include "basic_defs.h"
#include "databasics.h"
#include "solution.h"
#include "assert.h"
#include <cstdlib>
#include <inttypes.h>

#include <stdio.h>
#include <string.h>
#include <map>




void rebalance(const dist_sort_t *data, const dist_sort_size_t myDataCount, dist_sort_t **rebalancedData, dist_sort_size_t *rCount) {

	MPI_Barrier(MPI_COMM_WORLD);

	int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    dist_sort_size_t lpre_sum;
    dist_sort_size_t total_count;
    MPI_Scan(&myDataCount, &lpre_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&myDataCount, &total_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Status stats[2];
    MPI_Request reqs[2];


//    int avg_count = round(static_cast<double>(total_count)/static_cast<double>(comm_size));
    long long int avg_count = static_cast<long long int>(total_count/(comm_size));
    long int remain = total_count%comm_size;
    if (my_rank < remain) {
    	avg_count++;
    }

    long long int delta_pre = static_cast<long long int>(lpre_sum) - static_cast<long long int>(total_count/(comm_size)) * (static_cast<long long int>(my_rank) + 1);
    if (my_rank < remain) {
    	delta_pre -= (my_rank + 1);
    } else {
    	delta_pre -= remain;
    }

    long long int my_count = static_cast<long long int>(myDataCount);
    long long int delta_avg = my_count - avg_count;

    dist_sort_t * fwd_data_buff = NULL;
    long long int fwdCount = abs(delta_pre);
    bool fwdRecv = delta_pre < 0;

    dist_sort_t * bkwd_data_buff = NULL;
    long long int bkwdCount = abs(delta_avg - delta_pre);
    bool bkwdRecv = (delta_avg - delta_pre) < 0;

    *rebalancedData = (dist_sort_t*)malloc(sizeof(dist_sort_t)*(size_t)(avg_count));
    *rCount = 0;
    int curr_msg = 0;

    printf("Hello I'm Rank %d - My Cnt: %d, Avg Cnt: %lld, PreSum: %lld, dAvg: %lld, dPre: %lld, fwdCnt: %lld, bkwdCnt: %lld\n", my_rank, my_count, avg_count, lpre_sum, delta_avg, delta_pre, fwdCount, bkwdCount);
//    std::cout <<  "Hello I'm Rank" << my_rank << " - My Cnt: " << my_count << ", Avg Cnt: " << avg_count << ", PreSum: " << lpre_sum << ", dAvg: " << delta_avg << ", dPre: " << delta_pre << ", fwdCnt: " << fwdCount << ", bkwdCnt: " << bkwdCount << "\n";
    fflush(stdout);

    if (my_rank == comm_size - 1) {
    	assert(delta_pre == 0);
    	assert(fwdCount == 0);
    }
    if (my_rank == 0) {
    	assert(bkwdCount == 0);
    }

    //allocate memory
    if (fwdCount > 0) {
    	fwd_data_buff = (dist_sort_t*)malloc(sizeof(dist_sort_t)*(size_t)(fwdCount));
    }
    if (bkwdCount > 0) {
    	bkwd_data_buff = (dist_sort_t*)malloc(sizeof(dist_sort_t)*(size_t)(bkwdCount));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //recv data
    if (fwdCount > 0 && fwdRecv) {
    	//recv from next
    	assert(my_rank < comm_size - 1);
    	printf("Rank %d - RECV - from Rank %d - Cnt: %ld\n", my_rank, my_rank + 1, fwdCount);
    	fflush(stdout);
//    	std::cout <<  "Rank " << my_rank << " - RECV - from Rank " << my_rank + 1 << " - Cnt: " << fwdCount << "\n";
    	MPI_Irecv(&fwd_data_buff[0], static_cast<int>(fwdCount), MPI_UNSIGNED_LONG_LONG, my_rank + 1, 0, MPI_COMM_WORLD, &reqs[curr_msg]);
    	curr_msg++;
    }
    if (bkwdCount > 0 && bkwdRecv) {
    	//recv from prev
    	assert(my_rank > 0);
    	printf("Rank %d - RECV - from Rank %d - Cnt: %ld\n", my_rank, my_rank - 1, bkwdCount);
    	fflush(stdout);
//    	std::cout <<  "Rank " << my_rank << " - RECV - from Rank " << my_rank - 1 << " - Cnt: " << bkwdCount << "\n";
    	MPI_Irecv(&bkwd_data_buff[0], static_cast<int>(bkwdCount), MPI_UNSIGNED_LONG_LONG, my_rank - 1, 0, MPI_COMM_WORLD, &reqs[curr_msg]);
    	curr_msg++;
    }

    //send data
    if (fwdCount > 0 && !fwdRecv) {
    	//send to next
    	assert(my_rank < comm_size - 1);
    	if (fwdCount > myDataCount) {
    		MPI_Waitall(curr_msg, reqs, stats);
    	}

    	dist_sort_size_t rem = commit_send_data(data, &my_count, fwd_data_buff,  fwdCount);
    	if (rem > 0) {
    		assert(my_rank > 0);
    		rem = commit_send_data(bkwd_data_buff, &bkwdCount, fwd_data_buff,  rem);
    	}

    	printf("Rank %d - SEND - to Rank %d - Cnt: %ld\n", my_rank, my_rank + 1, fwdCount);
    	fflush(stdout);
//    	std::cout <<  "Rank " << my_rank << " - SEND - to Rank " << my_rank + 1 << " - Cnt: " << fwdCount << "\n";
		MPI_Isend(&fwd_data_buff[0], static_cast<int>(fwdCount), MPI_UNSIGNED_LONG_LONG, my_rank + 1, 0, MPI_COMM_WORLD, &reqs[curr_msg]);
		curr_msg++;
    }
    if (bkwdCount > 0 && !bkwdRecv) {
    	//send to prev
    	assert(my_rank > 0);
    	if (bkwdCount > myDataCount) {
    		MPI_Waitall(curr_msg, reqs, stats);
    	}

    	dist_sort_size_t rem = commit_send_data(data, &my_count, bkwd_data_buff,  bkwdCount);
    	if (rem > 0) {
    		assert(my_rank < comm_size - 1);
    		rem = commit_send_data(fwd_data_buff, &fwdCount, bkwd_data_buff,  rem);
    	}

    	printf("Rank %d - SEND - to Rank %d - Cnt: %ld\n", my_rank, my_rank - 1, bkwdCount);
    	fflush(stdout);
//    	std::cout <<  "Rank " << my_rank << " - SEND - to Rank " << my_rank - 1 << " - Cnt: " << bkwdCount << "\n";
		MPI_Isend(&bkwd_data_buff[0], static_cast<int>(bkwdCount), MPI_UNSIGNED_LONG_LONG, my_rank - 1, 0, MPI_COMM_WORLD, &reqs[curr_msg]);
		curr_msg++;
    }

    assert(curr_msg <= 2);
    MPI_Waitall(curr_msg, reqs, stats);

    MPI_Barrier(MPI_COMM_WORLD);
//    MPI_Barrier(MPI_COMM_WORLD);
//    delta_avg = my_count - avg_count;
//    printf("Hey I made it! I'm Rank %d - My Cnt: %d, Avg Cnt: %d, PreSum: %d, dAvg: %d, dPre: %d\n", my_rank, my_count, avg_count, lpre_sum, delta_avg, delta_pre);


    if (bkwdRecv) {
    	commit_data(bkwd_data_buff,  bkwdCount, *rebalancedData,  rCount);
    }
    if (fwdRecv) {
    	commit_data(fwd_data_buff,  fwdCount, *rebalancedData,  rCount);
    }
    commit_data(data,  my_count, *rebalancedData,  rCount);


    printf("All done! I'm Rank %d - myCnt: %llu, rCnt: %llu\n", my_rank, myDataCount, *rCount);
    fflush(stdout);
//    std::cout <<  "All done! I'm Rank " << my_rank << " - myCnt: " << myDataCount << ", rCnt: " << *rCount << "\n";


    MPI_Barrier(MPI_COMM_WORLD);


    dist_sort_size_t final_count;
    MPI_Allreduce(rCount, &final_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

//    printf("Rank %d - TotalCnt: %d, FinalCnt: %llu\n", my_rank, total_count, final_count);
    std::cout <<  "Rank " << my_rank << " - TotalCnt: " << total_count << ", FinalCnt: " << final_count << "\n";
    assert(final_count == (dist_sort_size_t)total_count);


    if (NULL != fwd_data_buff) free(fwd_data_buff);
	if (NULL != bkwd_data_buff) free(bkwd_data_buff);
}


void commit_data(const dist_sort_t * comm_buff,  long long int comm_buff_cnt, dist_sort_t * local_buff,  dist_sort_t * loc_buff_cnt) {
//	printf("Inside commit DATA - Committing: %d\n", comm_buff_cnt);
	for (long long int i = 0; i < comm_buff_cnt; i++) {
		local_buff[*loc_buff_cnt] = comm_buff[i];
		(*loc_buff_cnt)++;
	}
}


dist_sort_size_t commit_send_data(const dist_sort_t * read_buff, long long int * read_buff_cnt, dist_sort_t * comm_buff,  long long int comm_buff_cnt) {

	dist_sort_size_t j = 0;
//	printf("Inside commit SEND - Sending: %d\n", comm_buff_cnt);
	while(*read_buff_cnt > 0 && j < comm_buff_cnt) {
//		printf("Inside commit send - Sending: %d - Iteration: %d a - rBufCnt: %d\n", comm_buff_cnt, j, *read_buff_cnt);
		comm_buff[j] = read_buff[(*read_buff_cnt) - 1];
//		printf("Inside commit send - Sending: %d - Iteration: %d b - rBufCnt: %d\n", comm_buff_cnt, j, *read_buff_cnt);
		(*read_buff_cnt)--;
//		printf("Inside commit send - Sending: %d - Iteration: %d c - rBufCnt: %d\n", comm_buff_cnt, j, *read_buff_cnt);
		j++;
	}

	return comm_buff_cnt - j;
}
















//*\param data  Pointer to local data array. Immutable.
//*\param data_size  Size of local data array. Immutable.
//*\param splitters  Output. Array of size numSplitters containing the values that partition the data.
//*						The last one is the maximum value in the global data.
//*\param counts  Output. The total number, globally, of data falling into each bin.
//*   So counts[i] = COUNT(x s.t. splitter[i-1] < x <= splitter[i]) .
//*	( counts[0] = COUNT(x s.t. 0 <= x <= splitter[0]) )
//*\param numSplitters  The number of splitters being requested and the size of
//*						splitters and counts arrays. An arbitrary number >= 0.

void findSplitters(const dist_sort_t *data, const dist_sort_size_t data_size, dist_sort_t *splitters, dist_sort_size_t *counts, int numSplitters) {
	//N splitters for N bins == comm_size

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    dist_sort_size_t total_count;
    dist_sort_size_t my_data_size = data_size;

    MPI_Allreduce(&my_data_size, &total_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    long double target = (long double)total_count / (long double)numSplitters;
    long double tol = 0.01;


	dist_sort_size_t * sorted_data = (dist_sort_t*)malloc(sizeof(dist_sort_t)*(size_t)(data_size));
	dist_sort_size_t * my_counts = (dist_sort_t*)malloc(sizeof(dist_sort_t)*(size_t)(numSplitters));
	memcpy(sorted_data, data, sizeof(dist_sort_size_t)*data_size);


	// TODO: MAKE PARALLEL - reduce on my_max
	dist_sort_size_t my_max = 0;
	dist_sort_size_t my_min;
	for (dist_sort_size_t i = 0; i < data_size; i++) {
		assert(sorted_data[i] == data[i]);
		if (data[i] > my_max) {
			my_max = data[i];
		}

		if (i == 0) {
			my_min = data[i];
		} else if (data[i] < my_min) {
			my_min = data[i];
		}

	}

	sort(sorted_data, data_size);

    dist_sort_size_t global_max = 0;
    MPI_Allreduce(&my_max, &global_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    dist_sort_size_t global_min = 0;
    MPI_Allreduce(&my_min, &global_min, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);

//    printf("Hello from Rank %d - Cnt: %llu, MyCnt: %llu, GlobCnt: %llu, MyMax: %llu, GlobMax: %llu, Tgt: %Le, Tol: %Le, MyMin: %llu, GlobMin: %llu\n",
//    		my_rank, data_size, my_data_size, total_count, my_max, global_max, target, tol, my_min, global_min);

    MPI_Barrier(MPI_COMM_WORLD);

    int phase = 0;
    int keep_going = 1;
    do {
    	// calculate new splitters at rank 0
    	if (phase == 0 && my_rank == 0) {
    		init_splitters(splitters, &global_max, &global_min, &numSplitters);
    	} else if (my_rank == 0) {
    		recalc_splitters(splitters, counts, &numSplitters, &global_min);
    	}

    	// beoadcast out to homies
    	MPI_Bcast(&splitters[0], numSplitters, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

//    	if(my_rank == 0) {
//    		printf("Rank %d - Phase %d - Splitters:", my_rank, phase);
//    		print_array(splitters, &numSplitters);
//    	}

    	// everyone counts how many of each they have in counts
    	count_up(sorted_data, &data_size, splitters, &numSplitters, my_counts);

    	// reduce sum back to rank 0
    	MPI_Reduce(&my_counts[0], &counts[0], numSplitters, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    	if (my_rank == 0) {
    		keep_going = check_counts(counts, &target, &tol, &numSplitters, &phase);
    	}

    	MPI_Bcast(&keep_going, 1, MPI_INT, 0, MPI_COMM_WORLD);

//    	if (my_rank == 0) {
//        	printf("Rank %d - Phase %d - GlobalCounts:", my_rank, phase);
//        	print_array(counts, &numSplitters);
//    	}

//    	MPI_Barrier(MPI_COMM_WORLD);

    	phase++;

    } while (keep_going == 1);

//    printf("Rank %d - Yayyy I'm out!!!\n", my_rank);

    MPI_Bcast(&counts[0], numSplitters, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

//    printf("Counts found!! Goodbye from Rank %d - Counts:", my_rank);
//    print_array(counts, &numSplitters);
//
//    MPI_Barrier(MPI_COMM_WORLD);

//    if (my_rank == 0) {
//		printf("Splitters found!! Goodbye from Rank %d - Splitters:", my_rank);
//		print_array(splitters, &numSplitters);
//    }

    if (NULL != sorted_data) free(sorted_data);
	if (NULL != my_counts) free(my_counts);

	MPI_Barrier(MPI_COMM_WORLD);

}



void print_array(dist_sort_t * splitters, int * numSplitters) {
	for(int k = 0; k < *numSplitters; k++) {
		printf(" %llu", splitters[k]);
	}
	printf("\n");
}


void init_splitters(dist_sort_t * splitters, dist_sort_size_t * global_max, dist_sort_size_t * global_min, int * numSplitters) {

	for (int i = 0; i < *numSplitters - 1; i++) {
        splitters[i] = *global_min + (i+1)*((*global_max - *global_min)/(*numSplitters));
    }

	splitters[*numSplitters - 1] = *global_max;
//	printf("Rank 0 - I have initialized the Splitters:");
//	print_array(splitters, numSplitters);

}


void recalc_splitters(dist_sort_t * splitters, dist_sort_size_t *counts, int * numSplitters, dist_sort_size_t * global_min) {
    bool change = false;
    long double prev = *global_min;
    long double temp = 0;

	for (int i = 0; i < *numSplitters - 1; i++) {
		temp = splitters[i];
        splitters[i] = prev + ((long double)splitters[i] - prev)/(2 * (long double)counts[i])*((long double)counts[i] + (long double)counts[i+1]);
        prev = temp;
    }
}



void count_up(dist_sort_t *sorted_data, const dist_sort_size_t * data_size, const dist_sort_t *const splitters, int * numSplitters, dist_sort_size_t *my_counts) {

	for(int w = 0; w < *numSplitters; w++) {
		my_counts[w] = 0;
	}

	// TODO: MAKE PARALLEL - incrementation must be ATOMIC
	int curr_splitter = 0;
	for (int i = 0; i < *data_size; i++) {
		while (sorted_data[i] > splitters[curr_splitter]) {
			curr_splitter++;
			assert(curr_splitter < *numSplitters);
		}
		my_counts[curr_splitter]++;
	}

//	dist_sort_size_t check_sum = 0;
//
//	for(int w = 0; w < *numSplitters; w++) {
//		check_sum += my_counts[w];
//	}
}


int check_counts(dist_sort_size_t *counts, long double * target, long double * tol, int * numSplitters, int * phase) {

	int success = 0;
//	printf("Phase %d Deviations -", *phase);

	for (int i = 0; i < *numSplitters; i++) {
		long double bin_deviation = fabs(counts[i] - *target)/(*target);
//		printf(" %Le", bin_deviation);
		if(bin_deviation > *tol) {
			success = 1;
			break;
		}
	}
//	printf("\n");
	return success;
}



void sort(dist_sort_t *data, const dist_sort_size_t size) {
	// You are welcome to use this sort function.
	// If you wish to implement your own, you can do that too.
	// Don't use bubblesort.
	std::sort(data,data+size);
}









// Move data to the corresponding ranks
/**
*Move data to the appropriate machine/bucket according to a partitioning of data.
*
*
*\param sendData  Pointer to the local array of data to be redistributed. Immutable.
*\param sDataCount  Number of elements in sendData.
*\param recvData  Output. The address of the new, rebalanced array, will go in the
*					address pointed to by this pointer. Allocated with malloc().
*						Will only contain values in the range
*							(splitters[MY_RANK-1] < x <= splitters[MY_RANK] ].
*						( Rank 0 will get all data x <= splitters[0] ).
*\param rDataCount  Output. The size of the new, rebalanced array,
*						will be recorded at the address pointed to
*						by this parameter.
*
*\param splitters	Input. Array of size numSplitters containing the values
*							that partition the data.
*						The last one is the maximum value in the global data.
*							Ranks other than 0 should disregard this input and
*						use the value provided to rank 0.
*\param counts	Input. The total number, globally, of data falling into each bin.
*			So counts[i] = COUNT(x s.t. splitter[i-1] < x <= splitter[i]) .
*				( counts[0] = COUNT(x s.t. 0 <= x <= splitter[0]) )
*							Ranks other than 0 should disregard this input and
*						use the value provided to rank 0.
*							It is recommended, but not required, that all ranks
*						disregard this and calculate the appropriate rank sizes
*						from the splitters and data only.
*							(Reducing dependence on the correctness
*						of findSplitters.)
*\param numSplitters	The number of splitters being provided and the size of
*						splitters and counts arrays. An arbitrary number >= 0.
*/


void moveData(const dist_sort_t *const sendData, const dist_sort_size_t sDataCount, dist_sort_t **recvData, dist_sort_size_t *rDataCount,
		      const dist_sort_t *const splitters, const dist_sort_t *const counts, int numSplitters) {

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

//    printf("Hello from Rank %d - 1\n",my_rank);

    // sort my send data
//    dist_sort_t * sorted_data = (dist_sort_t*)malloc(sizeof(dist_sort_t)*(size_t)(sDataCount));
//	memcpy(sorted_data, sendData, sizeof(dist_sort_size_t)*sDataCount);
//	sort(sorted_data, sDataCount);


	// broadcast from Rank 0 how many counts each bin has
    dist_sort_t * glob_spltters = (dist_sort_t*)malloc(sizeof(dist_sort_t)*(size_t)(numSplitters));
    if (my_rank == 0) {
    	memcpy(glob_spltters, splitters, sizeof(dist_sort_size_t)*numSplitters);
    }
    MPI_Bcast(&glob_spltters[0], numSplitters, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
//    MPI_Barrier(MPI_COMM_WORLD);
//    print_array(glob_spltters, &numSplitters);
//    MPI_Barrier(MPI_COMM_WORLD);


	// count how many items I need to send to each processor
	dist_sort_t * my_counts_by_proc = (dist_sort_t*)malloc(sizeof(dist_sort_t)*(size_t)(comm_size));
	count_up_by_proc(sendData, &sDataCount, glob_spltters, &numSplitters, my_counts_by_proc, &comm_size);


//	MPI_Barrier(MPI_COMM_WORLD);
//	printf("Hello from Rank %d - MyCntsByProc:", my_rank);
//	print_array(my_counts_by_proc, &comm_size);


	// broadcast from Rank 0 how many counts each bin has
    dist_sort_t * glob_bin_counts = (dist_sort_t*)malloc(sizeof(dist_sort_t)*(size_t)(numSplitters));
    if (my_rank == 0) {
    	memcpy(glob_bin_counts, counts, sizeof(dist_sort_size_t)*numSplitters);
    }
    MPI_Bcast(&glob_bin_counts[0], numSplitters, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

//    MPI_Barrier(MPI_COMM_WORLD);
//    print_array(glob_bin_counts, &numSplitters);

    // calculate how many data items I am expected to receive
    *rDataCount = calc_proc_data_count(glob_bin_counts, &my_rank, &comm_size, &numSplitters);
	*recvData = (dist_sort_t*)malloc(sizeof(dist_sort_t)*(size_t)(*rDataCount));

//	printf("Hello from Rank %d - SCnt: %llu, RCnt: %llu\n",my_rank, sDataCount, *rDataCount);

	// find out where my starting location is on each processor's public array
	dist_sort_t * my_glob_proc_starts = (dist_sort_t*)malloc(sizeof(dist_sort_t)*(size_t)(comm_size));

//	MPI_Barrier(MPI_COMM_WORLD);
//	printf("Hello from Rank %d - 5\n",my_rank);


	MPI_Scan(&my_counts_by_proc[0], &my_glob_proc_starts[0], comm_size, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

//	printf("Hello from Rank %d - 6\n",my_rank);

	for (int i = 0; i < numSplitters; i++) {
		my_glob_proc_starts[i] -= my_counts_by_proc[i];
	}


//	MPI_Barrier(MPI_COMM_WORLD);
//	printf("Hello from Rank %d - 7\n",my_rank);





	// distribute data to its owner

	p2p_msgs(sendData, *recvData, rDataCount, my_counts_by_proc, &comm_size);

//	onesided_msg(sendData, *recvData, rDataCount, my_counts_by_proc, my_glob_proc_starts, &comm_size);




//	MPI_Barrier(MPI_COMM_WORLD);
//	printf("Hello from Rank %d - 8\n",my_rank);

	// once I have sent and received all data, sort my local data
//	sort(*recvData, *rDataCount);

//	MPI_Barrier(MPI_COMM_WORLD);
//	printf("Hello from Rank %d - 9\n",my_rank);

//    if (NULL != sorted_data) free(sorted_data);
	if (NULL != my_counts_by_proc) free(my_counts_by_proc);
    if (NULL != glob_bin_counts) free(glob_bin_counts);
	if (NULL != my_glob_proc_starts) free(my_glob_proc_starts);
	if (NULL != glob_spltters) free(glob_spltters);

//	MPI_Barrier(MPI_COMM_WORLD);

}




int get_proc_num_bins(int my_rank, int comm_size, int numSplitters) {
    int num_bins = numSplitters/comm_size;
    if (my_rank < numSplitters%comm_size) {
    	num_bins++;
    }
    return num_bins;
}


int get_proc_bin_pre_sum(int my_rank, int comm_size, int numSplitters) {
    int proc_start = numSplitters/comm_size * (my_rank + 1);
    if (my_rank < numSplitters%comm_size) {
    	proc_start += (my_rank + 1);
    } else {
    	proc_start += numSplitters%comm_size;
    }
	return proc_start;
}


dist_sort_t calc_proc_data_count(dist_sort_t * glob_bin_counts, int * my_rank, int * comm_size, int * numSplitters) {
	int my_bins = get_proc_num_bins(*my_rank, *comm_size, *numSplitters);
	int my_start = get_proc_bin_pre_sum(*my_rank, *comm_size, *numSplitters) - my_bins;
	dist_sort_t my_data_count = 0;

	for (int i = my_start; i < my_start + my_bins; i++) {
		my_data_count += glob_bin_counts[i];
	}
	return my_data_count;
}



void count_up_by_proc(const dist_sort_t *const sorted_data, const dist_sort_size_t * data_size, const dist_sort_t *const splitters, int * numSplitters, dist_sort_size_t *my_counts_by_proc, int * comm_size) {

	for(int w = 0; w < *comm_size; w++) {
		my_counts_by_proc[w] = 0;
	}


//	MPI_Barrier(MPI_COMM_WORLD);
//    int my_rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


	// TODO: MAKE PARALLEL - incrementation must be ATOMIC
	int curr_splitter = 0;
	int curr_proc = 0;

	for (int i = 0; i < *data_size; i++) {
		while (sorted_data[i] > splitters[curr_splitter]) {

//			if (my_rank == 0) {
//				printf("Incrementing Splitter - DataVal: %llu, OldSplVal: %llu, NewSplVal: %llu\n", sorted_data[i], splitters[curr_splitter], splitters[curr_splitter + 1]);
//			}

			curr_splitter++;
			assert(curr_splitter < *numSplitters);

			while (curr_splitter > get_proc_bin_pre_sum(curr_proc, *comm_size, *numSplitters) - 1) {

//				if (my_rank == 0) {
//					printf("Incrementing Proc - CurrSpl: %d, OldProc: %d, NewProc: %d\n", curr_splitter, curr_proc, curr_proc + 1);
//				}

				curr_proc++;
				assert(curr_proc < *comm_size);
				assert(curr_splitter >= get_proc_bin_pre_sum(curr_proc, *comm_size, *numSplitters) - get_proc_num_bins(curr_proc, *comm_size, *numSplitters));
			}

		}
		my_counts_by_proc[curr_proc]++;
	}

//	dist_sort_size_t check_sum = 0;
//
//	for(int w = 0; w < *numSplitters; w++) {
//		check_sum += my_counts[w];
//	}
}










void p2p_msgs(const dist_sort_t *const sorted_data, dist_sort_t *recvData, dist_sort_size_t *rDataCount, dist_sort_t * my_counts_by_proc, int * comm_size) {

//	int my_rank;
//	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
//
//
//	MPI_Barrier(MPI_COMM_WORLD);
//    printf("Hello from P2P - Rank %d - 1\n", my_rank);

	dist_sort_t * my_recv_counts_by_proc = (dist_sort_t*)malloc(sizeof(dist_sort_t)*(size_t)(*comm_size));
	MPI_Alltoall(&my_counts_by_proc[0], 1, MPI_UNSIGNED_LONG_LONG, &my_recv_counts_by_proc[0], 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);


//	MPI_Barrier(MPI_COMM_WORLD);
//	print_array(my_recv_counts_by_proc, comm_size);
//    MPI_Barrier(MPI_COMM_WORLD);

	int exp_msgs = 0;
	for (int p = 0; p < *comm_size; p++) {
		if (my_counts_by_proc[p] > 0) {
			exp_msgs++;
		}
		if (my_recv_counts_by_proc[p] > 0) {
			exp_msgs++;
		}
	}

//	MPI_Barrier(MPI_COMM_WORLD);
//    printf("Hello from P2P - Rank %d - 2\n", my_rank);

    MPI_Status stats[exp_msgs];
    MPI_Request reqs[exp_msgs];

	dist_sort_t send_start = 0;
	dist_sort_t recv_start = 0;
	int curr_msg = 0;

	for (int p = 0; p < *comm_size; p++) {

		if (my_counts_by_proc[p] > 0) {

//			if (my_rank == 0) {
//				printf("Hello from P2P Loop - Rank %d - Sending to Rank %d - Cnt: %llu, Start: %llu\n", my_rank, p, my_counts_by_proc[p], send_start);
//			}

			MPI_Isend(&sorted_data[send_start], my_counts_by_proc[p], MPI_UNSIGNED_LONG_LONG, p, 0, MPI_COMM_WORLD, &reqs[curr_msg]);
			curr_msg++;
		}

		if (my_recv_counts_by_proc[p] > 0) {

//			if (my_rank == 0) {
//				printf("Hello from P2P Loop - Rank %d - Recving from Rank %d - Cnt: %llu, Start: %llu\n", my_rank, p, my_recv_counts_by_proc[p], recv_start);
//			}

			MPI_Irecv(&recvData[recv_start], my_recv_counts_by_proc[p], MPI_UNSIGNED_LONG_LONG, p, 0, MPI_COMM_WORLD, &reqs[curr_msg]);
			curr_msg++;
		}

		send_start += my_counts_by_proc[p];
		recv_start += my_recv_counts_by_proc[p];
	}

//	printf("Hello from P2P - Rank %d - 3\n", my_rank);

	assert(recv_start == *rDataCount);

	MPI_Waitall(curr_msg, reqs, stats);

	MPI_Barrier(MPI_COMM_WORLD);

	if (NULL != my_recv_counts_by_proc) free(my_recv_counts_by_proc);

//	printf("Goodbye from P2P\n");
}


void onesided_msg(const dist_sort_t *const sorted_data, dist_sort_t *recvData, dist_sort_size_t *rDataCount, dist_sort_t * my_counts_by_proc, dist_sort_t * my_glob_proc_starts, int * comm_size) {

//    val window_buffer[*rDataCount];
//	MPI_Barrier(MPI_COMM_WORLD);
//
//    int my_rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

//    printf("Hello from 1-sided - Rank %d - 1\n", my_rank);


    MPI_Win window;
    MPI_Win_create(&recvData[0], *rDataCount*sizeof(dist_sort_t), sizeof(dist_sort_t), MPI_INFO_NULL, MPI_COMM_WORLD, &window);
    MPI_Barrier(MPI_COMM_WORLD);

//    MPI_Barrier(MPI_COMM_WORLD);
//    printf("Hello from 1-sided - Rank %d - 2\n", my_rank);

	dist_sort_t local_start = 0;

	for (int p = 0; p < *comm_size; p++) {

//		printf("Hello from 1-sided Loop - Rank %d - Sending to Rank %d - Cnt: %llu\n", my_rank, p, my_counts_by_proc[p]);

		if (my_counts_by_proc[p] > 0) {

			MPI_Win_fence(0, window);

			MPI_Put(&sorted_data[local_start], my_counts_by_proc[p], MPI_UNSIGNED_LONG_LONG, p, my_glob_proc_starts[p], my_counts_by_proc[p], MPI_UNSIGNED_LONG_LONG, window);

			// Wait for the MPI_Put issued to complete before going any further
			MPI_Win_fence(0, window);
		}

	    local_start += my_counts_by_proc[p];
	}

//	MPI_Barrier(MPI_COMM_WORLD);
//	printf("Goodbye from 1-sided - Rank %d\n", my_rank);

	// Destroy the window
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Win_free(&window);

}











