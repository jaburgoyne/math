#include <test/unit/math/test_ad.hpp>
#include <stan/math.hpp>
#include <stan/math/mix.hpp>
#include <test/unit/math/laplace/laplace_utility.hpp>
#include <test/unit/math/laplace/aki_synth_data/x1.hpp>

#include <test/unit/math/rev/fun/util.hpp>

#include <gtest/gtest.h>
#include <vector>
/*
TEST_P(bernoulli_logit_phi_dim500, specialized_function_val_test) {
  using stan::math::laplace_marginal_bernoulli_logit_lpmf;
  using stan::math::laplace_marginal_tol_bernoulli_logit_lpmf;
  using stan::math::to_vector;
  using stan::math::var;
  using stan::math::test::flag_test;
  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();
  constexpr double tolerance = 1e-12;
  constexpr int max_num_steps = 1000;

  std::vector<int> n_samples = stan::math::rep_array(1, dim_theta);
  Eigen::VectorXd theta_0 = Eigen::VectorXd::Zero(dim_theta);
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim_theta);
  std::vector<double> delta;
  std::vector<int> delta_int;
  Eigen::Matrix<double, Eigen::Dynamic, 1> phi_dbl{{1.6, 1}};
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "[          ] [  INFO  ]"
                    << " Skipping test for hessian of size " << theta_0.size()
                    << " with hessian block size of " << hessian_block_size << std::endl;
  }
  using stan::math::test::sqr_exp_kernel_functor;
  double target = laplace_marginal_tol_bernoulli_logit_lpmf(
      y, n_samples, 0, sqr_exp_kernel_functor{},
      std::forward_as_tuple(x, phi_dbl(0), phi_dbl(1)), theta_0, tolerance,
              max_num_steps, hessian_block_size, solver_num,
              max_steps_line_search, nullptr);
  // Benchmark against gpstuff.
  constexpr double tol = 8e-4;
  EXPECT_NEAR(-195.368, target, tol);
}
*/
TEST_P(bernoulli_logit_phi_dim500, specialized_function_ad_test) {
  using stan::math::laplace_marginal_bernoulli_logit_lpmf;
  using stan::math::laplace_marginal_tol_bernoulli_logit_lpmf;
  using stan::math::to_vector;
  using stan::math::var;
  using stan::math::test::flag_test;
  static std::once_flag log_once;
  std::call_once(log_once, [] {
    JLOG().set_file("../laplace_bernoulli_phi_dim500.jsonl", false);
  });
  static std::atomic<int> run_counter{0};

  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();
  Eigen::VectorXd theta_0 = Eigen::VectorXd::Zero(dim_theta);
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "[          ] [  INFO  ]"
                    << " Skipping test for hessian of size " << theta_0.size()
                    << " with hessian block size of " << hessian_block_size << std::endl;
  }
  constexpr double tolerance = 1e-12;
  constexpr int max_num_steps = 1000;
  JLOG().init_builder("test", "bernoulli_logit_dim_500_" + std::to_string(solver_num) + "_" +
                                  std::to_string(hessian_block_size) + "_" +
                                  std::to_string(max_steps_line_search));

  std::vector<int> n_samples = stan::math::rep_array(1, dim_theta);
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim_theta);
  std::vector<double> delta;
  std::vector<int> delta_int;
  Eigen::Matrix<double, Eigen::Dynamic, 1> phi_dbl{{1.6, 1}};
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "[          ] [  INFO  ]"
                    << " Skipping test for hessian of size " << theta_0.size()
                    << " with hessian block size of " << hessian_block_size << std::endl;
  }
  using stan::math::test::sqr_exp_kernel_functor;
  constexpr stan::test::ad_tolerances tols{
      stan::test::ad_gradient_tols{1e-8, 1e-3}};
  auto f = [&](auto&& alpha, auto&& rho) {
    auto __b = JLOG().builder();
    __b.field("component", "bernoulli_logit_dim_500")
      .field("where", "param_test")
      .field("event", "laplace_marginal_tol_call")
      .field("v_level", 0)
      .field("run_num", ++run_counter)
      .begin_object("test")
        .field("solver_num", solver_num)
        .field("hessian_block_size", hessian_block_size)
        .field("max_steps_line_search", max_steps_line_search)
      .end()
      .begin_object("autodiff")
        .field("alpha",
               (bool)stan::is_any_autodiff_v<decltype(alpha)>)
        .field("rho",
               (bool)stan::is_any_autodiff_v<decltype(rho)>)
      .end();
      auto __t0 = std::chrono::high_resolution_clock::now();
      try {
        auto val = laplace_marginal_tol_bernoulli_logit_lpmf(
            y, n_samples, mean, sqr_exp_kernel_functor{},
            std::forward_as_tuple(x, alpha, rho), theta_0, tolerance,
            max_num_steps, hessian_block_size, solver_num,
            max_steps_line_search, nullptr);
        auto end_t0 = std::chrono::high_resolution_clock::now();
        auto __ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end_t0 - __t0)
                        .count();
        __b.field("v_ns", (long long)__ns);
        if (::testing::Test::HasNonfatalFailure()) {
          __b.field("status","FAILURE");
        } else {
          __b.field("status","SUCCESS");
        }
        JLOG().commit_now(JsonLogger::Level::Debug,
                          "bernoulli_logit_dim_500", __b);
        return val;
      } catch (const std::exception& e) {
        auto end_t0 = std::chrono::high_resolution_clock::now();
        auto __ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end_t0 - __t0)
                        .count();
        __b.field("v_ns", (long long)__ns);
        __b.field("status","FAILURE");
        __b.field("exception", e.what());
        JLOG().commit_now(JsonLogger::Level::Debug,
                          "bernoulli_logit_dim_500", __b);
        throw e;
      }
  };
  stan::test::expect_ad<true>(tols, f, phi_dbl[0], phi_dbl[1]);
}

INSTANTIATE_TEST_SUITE_P(
    BernoulliLogitPhi500MapParamTestSuite,
    bernoulli_logit_phi_dim500,
    ::testing::Combine(
        ::testing::Values(1, 2, 3),      // solver_num
        ::testing::Values(1, 2, 3),      // hessian_block_size
        ::testing::Values(0, 500, 1000)       // max_steps_line_search
    ),
    ParamName);
