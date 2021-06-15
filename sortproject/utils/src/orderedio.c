
#include "orderedio.h"

#include <mpi.h>

//Adapted from https://en.cppreference.com/w/c/variadic

#include <stdio.h>
#include <stdarg.h>

void mpi_ordered_printf(MPI_Comm comm, const char* fmt, ...)
{
	int commrank, commsize;
	MPI_Comm_size(comm, &commsize);
	MPI_Comm_rank(comm, &commrank);

	va_list args;
	va_start(args, fmt);

	for(int i = 0;i<commsize;++i){
		MPI_Barrier(comm);
		if(i == commrank){
			vprintf(fmt, args);
		}
		MPI_Barrier(comm);
	}

	va_end(args);
}
