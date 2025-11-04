#include <test/unit/pretty_print_types.hpp>
#include <test/unit/math/test_ad.hpp>
#include <stan/math.hpp>
#include <stan/math/mix.hpp>
#include <test/unit/math/laplace/laplace_utility.hpp>
#include <test/unit/math/rev/fun/util.hpp>

#include <gtest/gtest.h>
#include <vector>

TEST_P(LaplaceSolverGrid, laplace_marginal_neg_binomial_log_lpmf_phi_dim_2) {
  using stan::math::laplace_marginal_neg_binomial_2_log_lpmf;
  using stan::math::laplace_marginal_tol_neg_binomial_2_log_lpmf;
  using stan::math::to_vector;
  using stan::math::value_of;
  using stan::math::var;
  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();
  static std::once_flag log_once;
  std::call_once(log_once, [] {
    JLOG().set_file("../laplace_marginal_neg_binomial_log_lpmf_phi_dim_2.jsonl", false);
  });
  static std::atomic<int> run_counter{0};

  constexpr double alpha_dbl = 1.6;
  constexpr double rho_dbl = 0.45;
  constexpr int dim_theta = 2;
  Eigen::VectorXd theta_0{{0, 0}};
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "Skipping: theta0.size() = " << theta_0.size()
                 << " not divisible by hessian_block_size = "
                 << hessian_block_size;
  }
  std::vector<Eigen::VectorXd> x(dim_theta);
  Eigen::VectorXd x_0{{0.05100797, 0.16086164}};
  Eigen::VectorXd x_1{{-0.59823393, 0.98701425}};
  x[0] = x_0;
  x[1] = x_1;
  std::vector<int> y{1, 0};
  std::vector<int> y_index{1, 2};
  constexpr double eta_dbl = 100;
  constexpr double tolerance = 1e-12;
  constexpr int max_num_steps = 1000;
  constexpr stan::test::ad_tolerances tols{
      stan::test::ad_gradient_tols{1e-8, 1e-2}};
  JLOG().init_builder("test", "laplace_marginal_neg_binomial_2_" + std::to_string(solver_num) + "_" +
                                  std::to_string(hessian_block_size) + "_" +
                                  std::to_string(max_steps_line_search));

  auto f = [&](auto&& alpha, auto&& rho, auto&& eta) {
    auto __b = JLOG().builder();
    __b.field("component", "laplace_marginal_neg_binomial_2")
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
      .field("alpha", stan::is_any_autodiff_v<decltype(alpha)>)
      .field("rho", stan::is_any_autodiff_v<decltype(rho)>)
      .field("eta", stan::is_any_autodiff_v<decltype(eta)>)
    .end();
  auto __t0 = std::chrono::high_resolution_clock::now();
  try {
    return laplace_marginal_tol_neg_binomial_2_log_lpmf(
        y, y_index, eta, 0, stan::math::test::squared_kernel_functor{},
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
                        "laplace_marginal_neg_binomial_2", __b);
    } catch (const std::exception& e) {
      auto end_t0 = std::chrono::high_resolution_clock::now();
      auto __ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      end_t0 - __t0)
                      .count();
      __b.field("error", e.what());
      __b.field("v_ns", (long long)__ns);
      if (::testing::Test::HasNonfatalFailure()) {
        __b.field("status","FAILURE");
      } else {
        __b.field("status","SUCCESS");
      }
      JLOG().commit_now(JsonLogger::Level::Debug,
                        "laplace_marginal_neg_binomial_2", __b);
      throw e;
    }
  };
  stan::test::expect_ad<true>(tols, f, alpha_dbl, rho_dbl, eta_dbl);
}

INSTANTIATE_TEST_SUITE_P(
    NegBinomial2LogMapParamTestSuite,
    LaplaceSolverGrid,
    ::testing::Combine(
        ::testing::Values(1, 2, 3),      // solver_num
        ::testing::Values(1, 2, 3),      // hessian_block_size
        ::testing::Values(0, 250)       // max_steps_line_search
    ),
    ParamName);

TEST_P(laplace_disease_map_test, laplace_marginal_neg_binomial_2_log_lpmf) {
  using stan::is_var_v;
  using stan::math::laplace_marginal_neg_binomial_2_log_lpmf;
  using stan::math::laplace_marginal_tol_neg_binomial_2_log_lpmf;
  using stan::math::to_vector;
  using stan::math::value_of;
  using stan::math::var;
  static std::atomic<int> run_counter{0};
  constexpr double eta_val = 1;
  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "Skipping: theta0.size() = " << theta_0.size()
                 << " not divisible by hessian_block_size = "
                 << hessian_block_size;
  }
  JLOG().init_builder("test", "disease_map_marginal_neg_binomial_2_" + std::to_string(solver_num) + "_" +
                                  std::to_string(hessian_block_size) + "_" +
                                  std::to_string(max_steps_line_search));
/*
  double marginal_density = laplace_marginal_neg_binomial_2_log_lpmf(
      y, y_index, eta_val, mean, stan::math::test::sqr_exp_kernel_functor(),
      std::forward_as_tuple(x, phi_dbl(0), phi_dbl(1)), nullptr);
*/
  // TODO(charlesm93): get benchmark from GPStuff or another software.
  constexpr double tolerance = 1e-12;
  constexpr int max_num_steps = 100;
  auto f = [&](auto&& alpha, auto&& rho, auto&& eta) {
    auto __b = JLOG().builder();
    __b.field("component", "disease_map_marginal_neg_binomial_2")
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
      .field("alpha", stan::is_any_autodiff_v<decltype(alpha)>)
      .field("rho", stan::is_any_autodiff_v<decltype(rho)>)
      .field("eta", stan::is_any_autodiff_v<decltype(eta)>)
    .end();
  auto __t0 = std::chrono::high_resolution_clock::now();
  try {
    auto val = laplace_marginal_tol_neg_binomial_2_log_lpmf(
        y, y_index, eta, mean, stan::math::test::sqr_exp_kernel_functor{},
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
                        "disease_map_marginal_neg_binomial_2", __b);
      return val;
    } catch (const std::exception& e) {
      auto end_t0 = std::chrono::high_resolution_clock::now();
      auto __ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      end_t0 - __t0)
                      .count();
      __b.field("error", e.what());
      __b.field("v_ns", (long long)__ns);
      if (::testing::Test::HasNonfatalFailure()) {
        __b.field("status","FAILURE");
      } else {
        __b.field("status","SUCCESS");
      }
      JLOG().commit_now(JsonLogger::Level::Debug,
                        "disease_map_marginal_neg_binomial_2", __b);
      throw e;
    }

  };
  auto ret = f(phi_dbl[0], phi_dbl[1], eta_val);
  stan::test::expect_ad<true>(f, phi_dbl[0], phi_dbl[1], eta_val);
}

INSTANTIATE_TEST_SUITE_P(
    NegBinomial2LogMapParamTestSuite,
    laplace_disease_map_test,
    ::testing::Combine(
        ::testing::Values(1, 2, 3),      // solver_num
        ::testing::Values(1, 2, 3),      // hessian_block_size
        ::testing::Values(0, 250)       // max_steps_line_search
    ),
    ParamName);
