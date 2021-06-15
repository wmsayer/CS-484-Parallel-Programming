#ifndef DATAGENERATION_H
#define DATAGENERATION_H

#include "basic_defs.h"

dist_sort_size_t chooseArraySize(int dist_choice, dist_sort_size_t a, dist_sort_size_t b);

dist_sort_size_t chooseArraySize_0(dist_sort_size_t a, dist_sort_size_t b);
dist_sort_size_t chooseArraySize_1(dist_sort_size_t a, dist_sort_size_t b);
dist_sort_size_t chooseArraySize_2(dist_sort_size_t a, dist_sort_size_t b);


void generateData(int dist_choice, dist_sort_t **data, dist_sort_size_t size, dist_sort_size_t min, dist_sort_size_t max);

void generateData_0(dist_sort_t *data, dist_sort_size_t size, dist_sort_size_t min, dist_sort_size_t max);
void generateData_1(dist_sort_t *data, dist_sort_size_t size, dist_sort_size_t min, dist_sort_size_t max);
void generateData_2(dist_sort_t *data, dist_sort_size_t size, dist_sort_size_t min, dist_sort_size_t max);
void generateData_3(dist_sort_t *data, dist_sort_size_t size, dist_sort_size_t min, dist_sort_size_t max);


#endif /*DATAGENERATION_H*/
