/*******************************************************************************
 *
 * Copyright (c) 2016-2018, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory
 * Written by Geoffrey M. Oxberry <oxberry1@llnl.gov>
 * LLNL-CODE-739313
 * All rights reserved.
 *
 * This file is part of gtest-mpi-listener. For details, see
 * https://github.com/LLNL/gtest-mpi-listener
 *
 * Please also see the LICENSE and COPYRIGHT files for the BSD-3
 * license and copyright information.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the disclaimer below.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the disclaimer (as noted below)
 *   in the documentation and/or other materials provided with the
 *   distribution.
 *
 * * Neither the name of the LLNS/LLNL nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
 * LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
*******************************************************************************/

#include "gtest/gtest.h"
#include "gtest-mpi-listener.hpp"
#include "mpi.h"

#include "databasics.h"
// Simple-minded functions for some testing


TEST(DATA, CheckGenerator) {
	uint64_t accumulator;
	for(int i = 0;i<1000;i++){
		accumulator |= randuint64();
	}//Could fail even if the generator is good, but the probability is (1/2)^1000.
	//maybe less since the numbers are pseudo-random and we take 1000 of them in order.

	EXPECT_EQ(accumulator, 0xffffffffffffffff);

}

// These tests could be made shorter with a fixture, but a fixture
// deliberately isn't used in order to make the test harness extremely simple


TEST(DATA, CheckChecksum) {
	const int NUM_REPEATS = 5;
	const int PERMUTE_NUMBER = 20;
	uint64_t individual_sums[NUM_REPEATS];

	srand(time(0));

	uint64_t confirm_mask = randuint64();
	uint64_t confirm_modulus = randuint64() % 0xffffffff;

	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Barrier(MPI_COMM_WORLD);

	uint64_t *randdata;
	uint64_t *assignments;
	uint64_t *assign_totals;
	uint64_t *mydata;

	int gen_size = size*7;

	//It doesn't have to be efficient. It just has to be correct.

	randdata = (uint64_t*)malloc(sizeof(uint64_t)*gen_size);
	assignments = (uint64_t*)malloc(sizeof(uint64_t)*gen_size);
	assign_totals = (uint64_t*)malloc(sizeof(uint64_t)*gen_size);

	if(0 == rank){
		for(int i = 0;i<gen_size;i++){
			randdata[i] = randuint64();
		}
	}



	for(int j = 0;j<NUM_REPEATS;j++){
		//Randomly reassign
		if(0 == rank){
			for(int i = 0;i<size;i++){
				assign_totals[i] = 0;
			}

			for(int i = 0;i<gen_size;i++){
				uint64_t ass = randuint64() % size;
				assignments[i] = ass;
				assign_totals[ass] += 1;
			}

			for(int i = 0;i < PERMUTE_NUMBER;i++){
				int index_a = rand() % gen_size;
				int index_b = rand() % gen_size;

				uint64_t a = randdata[index_a];
				randdata[index_a] = randdata[index_b];
				randdata[index_b] = a;
			}
		}

		//Here because we permuted the data on the leader.
		MPI_Bcast(randdata,gen_size,MPI_UNSIGNED_LONG_LONG,0,MPI_COMM_WORLD);
		MPI_Bcast(assignments,gen_size,MPI_UNSIGNED_LONG_LONG,0,MPI_COMM_WORLD);
		MPI_Bcast(assign_totals,size,MPI_UNSIGNED_LONG_LONG,0,MPI_COMM_WORLD);


		mydata = (uint64_t*)malloc(sizeof(uint64_t)*assign_totals[rank]);

		int k = 0;
		for(int i = 0;i<gen_size;i++){
			if(assignments[i] == rank)
				mydata[k++] = randdata[i];
		}

		uint64_t one_checksum = distributed_checksum(mydata,assign_totals[rank],confirm_mask, confirm_modulus, MPI_COMM_WORLD);
		individual_sums[j] = one_checksum;

		free(mydata);
	} // End for NUM_REPEATS

	free(randdata);
	free(assignments);
	free(assign_totals);

	for(int i = 1;i<NUM_REPEATS;++i){
		EXPECT_EQ(individual_sums[i-1], individual_sums[i]);
	}


	MPI_Barrier(MPI_COMM_WORLD);
}




int main(int argc, char** argv) {
  // Filter out Google Test arguments
  ::testing::InitGoogleTest(&argc, argv);

  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Add object that will finalize MPI on exit; Google Test owns this pointer
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);

  // Get the event listener list.
  ::testing::TestEventListeners& listeners =
      ::testing::UnitTest::GetInstance()->listeners();

  // Remove default listener
  delete listeners.Release(listeners.default_result_printer());

  // Adds MPI listener; Google Test owns this pointer
  listeners.Append(new MPIMinimalistPrinter);

  // Run tests, then clean up and exit
  int run_all_ret = RUN_ALL_TESTS();

  return run_all_ret;
}
