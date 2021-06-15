#ifndef _ORDEREDIO_H_
#define _ORDEREDIO_H_

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif
//Adapted from https://en.cppreference.com/w/c/variadic

void mpi_ordered_printf(MPI_Comm comm, const char* fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* _ORDEREDIO_H_ */
