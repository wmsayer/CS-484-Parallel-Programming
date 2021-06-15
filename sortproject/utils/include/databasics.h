#ifndef DATABASICS_H
#define DATABASICS_H

#include <stdint.h>

#include <mpi.h>

uint64_t data_randsize(double min, double max, double mean);

/**
*Returns a random uint64_t where every bit is a random bit.
*
*/
uint64_t randuint64();

uint64_t randuint64(uint64_t min, uint64_t max);

void allocate_and_populate(uint64_t **data, const uint64_t size);

uint64_t local_checksum(const uint64_t *data,
	const uint64_t my_size,
	const uint64_t mask, const uint64_t modulus);

uint64_t distributed_checksum(const uint64_t *data,
	const uint64_t my_size,
	const uint64_t mask, const uint64_t modulus,
	MPI_Comm comm);

#endif /*DATABASICS_H*/
