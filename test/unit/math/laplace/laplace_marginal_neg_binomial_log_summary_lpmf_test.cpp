#include <test/unit/pretty_print_types.hpp>
#include <test/unit/math/test_ad.hpp>
#include <stan/math.hpp>
#include <stan/math/mix.hpp>
#include <test/unit/math/laplace/laplace_utility.hpp>
#include <test/unit/math/rev/fun/util.hpp>

#include <gtest/gtest.h>
#include <vector>

TEST_P(LaplaceSolverGrid, laplace_marginal_neg_binomial_log_summary_lpmf_phi_dim_2) {
  using stan::math::laplace_marginal_neg_binomial_2_log_summary_lpmf;
  using stan::math::laplace_marginal_tol_neg_binomial_2_log_summary_lpmf;
  using stan::math::to_vector;
  using stan::math::value_of;
  using stan::math::var;

  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();
  Eigen::VectorXd theta_0{{0.5, 0.5}};  // moves s toward 0.5
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "[          ] [  INFO  ]"
                    << " Skipping test for hessian of size " << theta_0.size()
                    << " with hessian block size of " << hessian_block_size << std::endl;
  }

  constexpr double alpha_dbl = 3.0;     // stronger mean sensitivity
  constexpr double rho_dbl = 0.25;      // sharper lengthscale
  constexpr int dim_theta = 2;

  std::vector<Eigen::VectorXd> x(dim_theta);
  Eigen::VectorXd x_0{{0.0, 0.0}};
  Eigen::VectorXd x_1{{2.0, 2.0}};
  x[0] = x_0;
  x[1] = x_1;
  std::vector<int> y{3, 2, 4, 1, 0, 1, 0, 2};
  std::vector<int> y_index{1, 1, 1, 1,
                           2, 2, 2, 2};  // 4 obs/group → bigger n, counts
  constexpr double eta_dbl = 1.0;        // puts s ~ 0.5
  std::vector<int> n_per_group(theta_0.size(), 0);
  std::vector<int> counts_per_group(theta_0.size(), 0);
  for (int i = 0; i < y.size(); i++) {
    n_per_group[y_index[i] - 1]++;
    counts_per_group[y_index[i] - 1] += y[i];
  }
  constexpr double tolerance = 1e-12;
  constexpr int max_num_steps = 1000;
  constexpr stan::test::ad_tolerances tols{
      stan::test::ad_gradient_tols{1e-8, 1e-4}};
  auto f = [&](auto&& alpha, auto&& rho, auto&& eta) {
    return laplace_marginal_tol_neg_binomial_2_log_summary_lpmf(
        y, n_per_group, counts_per_group, eta, 0,
        stan::math::test::squared_kernel_functor{},
        std::forward_as_tuple(x, alpha, rho), theta_0, tolerance,
        max_num_steps, hessian_block_size, solver_num,
        max_steps_line_search, nullptr);
  };
  stan::test::expect_ad<true>(f, alpha_dbl, rho_dbl, eta_dbl);
}

INSTANTIATE_TEST_SUITE_P(
    NegBinomial2LogSummaryMapParamTestSuite,
    LaplaceSolverGrid,
    ::testing::Combine(
        ::testing::Values(1, 2, 3),      // solver_num
        ::testing::Values(1, 2, 3),      // hessian_block_size
        ::testing::Values(0, 250)       // max_steps_line_search
    ),
    ParamName);

TEST_P(laplace_disease_map_test,
       laplace_marginal_neg_binomial_2_log_summary_lpmf) {
  using stan::is_var_v;
  using stan::math::laplace_marginal_neg_binomial_2_log_summary_lpmf;
  using stan::math::laplace_marginal_tol_neg_binomial_2_log_summary_lpmf;
  using stan::math::to_vector;
  using stan::math::value_of;
  using stan::math::var;
  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "[          ] [  INFO  ]"
                    << " Skipping test for hessian of size " << theta_0.size()
                    << " with hessian block size of " << hessian_block_size << std::endl;
  }
  constexpr double eta = 1;
  std::vector<int> n_per_group(theta_0.size(), 0);
  std::vector<int> counts_per_group(theta_0.size(), 0);
  for (int i = 0; i < y.size(); i++) {
    n_per_group[y_index[i] - 1]++;
    counts_per_group[y_index[i] - 1] += y[i];
  }

  double marginal_density = laplace_marginal_neg_binomial_2_log_summary_lpmf(
      y, n_per_group, counts_per_group, eta, mean,
      stan::math::test::sqr_exp_kernel_functor{},
      std::forward_as_tuple(x, phi_dbl(0), phi_dbl(1)), nullptr);

  // ToDo (charlesm93): get benchmark from GPStuff or another software.
  constexpr double tolerance = 1e-12;
  constexpr int max_num_steps = 100;
  auto f = [&](auto&& alpha, auto&& rho, auto&& eta) {
    return laplace_marginal_tol_neg_binomial_2_log_summary_lpmf(
        y, n_per_group, counts_per_group, eta, 0,
        stan::math::test::sqr_exp_kernel_functor{},
        std::forward_as_tuple(x, alpha, rho), theta_0, tolerance,
        max_num_steps, hessian_block_size, solver_num,
        max_steps_line_search, nullptr);
  };
  auto ret = f(phi_dbl[0], phi_dbl[1], eta);
  stan::test::expect_ad<true>(f, phi_dbl[0], phi_dbl[1], eta);
}

INSTANTIATE_TEST_SUITE_P(
    NegBinomial2LogSummaryMapParamTestSuite,
    laplace_disease_map_test,
    ::testing::Combine(
        ::testing::Values(1, 2, 3),      // solver_num
        ::testing::Values(1, 2, 3),      // hessian_block_size
        ::testing::Values(0, 250)       // max_steps_line_search
    ),
    ParamName);
