#ifndef SORT_SOLUTION_H
#define SORT_SOLUTION_H

#include "basic_defs.h"

// Rebalance the data so that all ranks have the same number of elements
/**
*Redistribute data so that it is closer to balanced.
*
*This function redistributes data from nodes with more data to nodes with less.
* In the end, all nodes will have between 0.99 and 1.01 of the average data size.
* The rebalanced data will be placed into a new array allocated with malloc()
*  and a pointer to that new array will be returned via the return parameters.
*
*\param data  Pointer to local data array before rebalance. Is not altered.
*\param myDataCount  Size of local data array before rebalance.
*\param rebalancedData  Output. The address of the new, rebalanced array, will go in the
					address pointed to by this pointer. Allocated with malloc().
*\param rCount  Output. The size of the new, rebalanced array, will be recorded at the
					address pointed to by this parameter.
*
*/
void rebalance(const dist_sort_t *const data, const dist_sort_size_t myDataCount,
    dist_sort_t **rebalancedData, dist_sort_size_t *rCount);

void commit_data(const dist_sort_t * comm_buff,  long long int comm_buff_cnt, dist_sort_t * local_buff,  dist_sort_t * loc_buff_cnt);
dist_sort_size_t commit_send_data(const dist_sort_t * read_buff, long long int * read_buff_cnt, dist_sort_t * comm_buff,  long long int comm_buff_cnt);

// Perform histogramming to figure out data ranges for all ranks
/**
*Use distributed histogramming to find roughly even data ranges.
*
*This function finds a set of splitters that partition the global data into bins
* of roughly equal size. The bin sizes will be within 1% of
* TOTAL_GLOBAL_DATA_SIZE/NUM_RANKS .
* Bin i will contain all of the data s.t. splitters[i-1] < x <= splitters[i] .
* The implicit start of the first bin (implicit splitters[-1]) shall be 0.
* The last splitter will be the global maximum value over all data.
*
*\param data  Pointer to local data array. Immutable.
*\param data_size  Size of local data array. Immutable.
*\param splitters  Output. Array of size numSplitters containing the values that partition the data.
*						The last one is the maximum value in the global data.
*\param counts  Output. The total number, globally, of data falling into each bin.
*   So counts[i] = COUNT(x s.t. splitter[i-1] < x <= splitter[i]) .
*	( counts[0] = COUNT(x s.t. 0 <= x <= splitter[0]) )
*\param numSplitters  The number of splitters being requested and the size of
*						splitters and counts arrays. An arbitrary number >= 0.
*/
void findSplitters(const dist_sort_t *const data, const dist_sort_size_t data_size,
    dist_sort_t *splitters, dist_sort_size_t *counts, const int numSplitters);
void init_splitters(dist_sort_t * splitters, dist_sort_size_t * global_min, dist_sort_size_t * global_max, int * numSplitters);
void recalc_splitters(dist_sort_t * splitters, dist_sort_size_t *counts, int * numSplitters, dist_sort_size_t * global_min);
void count_up(dist_sort_t *sorted_data, const dist_sort_size_t * data_size, const dist_sort_t *const splitters, int * numSplitters, dist_sort_size_t *my_counts);
int check_counts(dist_sort_size_t *counts, long double * target, long double * tol, int * numSplitters, int * phase);
void print_array(dist_sort_t * splitters, int * numSplitters);

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
void moveData(const dist_sort_t *const sendData, const dist_sort_size_t sDataCount,
    dist_sort_t **recvData, dist_sort_size_t *rDataCount,
    const dist_sort_t *const splitters, const dist_sort_t *const counts, const int numSplitters);

dist_sort_t calc_proc_data_count(dist_sort_t * glob_counts, int * my_rank, int * comm_size, int * numSplitters);
void count_up_by_proc(const dist_sort_t *const sorted_data, const dist_sort_size_t * data_size, const dist_sort_t *const splitters, int * numSplitters, dist_sort_size_t *my_counts_by_proc, int * comm_size);
void p2p_msgs(const dist_sort_t *const sorted_data, dist_sort_t *recvData, dist_sort_size_t *rDataCount, dist_sort_t * my_counts_by_proc, int * comm_size);
void onesided_msg(const dist_sort_t *const sorted_data, dist_sort_t *recvData, dist_sort_size_t *rDataCount, dist_sort_t * my_counts, dist_sort_t * my_glob_starts, int * comm_size);

// Local sorting function
/**
*Sorts an array of type dist_sort_t from least to greatest.
*
*\param data  Pointer to an array of dist_sort_t to be sorted.
*\param size  Number of elements in the array to sort.
*/
void sort(dist_sort_t *data, const dist_sort_size_t size);

#endif /*SORT_SOLUTION_H*/
