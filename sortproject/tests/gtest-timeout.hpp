#ifndef _GTEST_TIMEOUT_HPP_
#define _GTEST_TIMEOUT_HPP_

//
// Code taken verbatim from example at http://antonlipov.blogspot.com/2015/08/how-to-timeout-tests-in-gtest.html
//
// See also : https://gist.github.com/compomega/4f59aee81bb03e2376d5421ca6282b67
//

/*
#include <future>

#define TEST_TIMEOUT_BEGIN   std::promise<bool> promisedFinished; \
	auto futureResult = promisedFinished.get_future(); \
	std::thread([&](std::promise<bool>& finished) {



#define TEST_TIMEOUT_FAIL_END(X)  finished.set_value(true); \
	}, std::ref(promisedFinished)).detach(); \
	EXPECT_TRUE(futureResult.wait_for(std::chrono::milliseconds(X)) != std::future_status::timeout) << "Timed Out.";



#define TEST_TIMEOUT_SUCCESS_END(X)  finished.set_value(true); \
	}, std::ref(promisedFinished)).detach(); \
	EXPECT_FALSE(futureResult.wait_for(std::chrono::milliseconds(X)) != std::future_status::timeout) << "Finished too soon.";
*/

#define TEST_TIMEOUT_BEGIN
#define TEST_TIMEOUT_FAIL_END(X)
#define TEST_TIMEOUT_SUCCESS_END(X)

#endif // _GTEST_TIMEOUT_HPP_
