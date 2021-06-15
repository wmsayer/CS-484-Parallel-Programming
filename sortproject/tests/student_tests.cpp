#include <random>

#include <gtest/gtest.h>
#include "gtest-mpi-listener.hpp"
#include "mpi.h"


#include "databasics.h"
#include "datageneration.h"
#include "basic_defs.h"
#include "solution.h"

#include <string>
#include <sstream>


//You can build your own tests for any of your own functions or helpers and put them here.

TEST(STUDENT, student_test_1){

	EXPECT_LT(1,2) << "We expected 1 to be less than 2, this is the error message printed";
	ASSERT_GE(4,3) << "Assert finishes the test immediately?";
	EXPECT_LE(2,2);
	EXPECT_EQ(true,true);
}

TEST(STUDENT, student_test_2){

	EXPECT_EQ(true,true);
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
