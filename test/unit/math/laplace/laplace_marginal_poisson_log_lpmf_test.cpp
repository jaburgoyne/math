#include <test/unit/math/test_ad.hpp>
#include <stan/math.hpp>
#include <stan/math/mix.hpp>
#include <test/unit/math/laplace/laplace_utility.hpp>
#include <test/unit/math/rev/fun/util.hpp>

#include <gtest/gtest.h>
#include <vector>

TEST_P(LaplaceSolverGrid, laplace_marginal_poisson_log_lpmf_phi_dim_2) {
  static std::once_flag log_once;
  std::call_once(log_once, [] {
    JLOG().set_file("../laplace_poisson_log_lpmf_phi_dim_2.jsonl", false);
  });
  static std::atomic<int> run_counter{0};

  using stan::math::laplace_marginal_poisson_log_lpmf;
  using stan::math::laplace_marginal_tol_poisson_log_lpmf;

  using stan::math::log;
  using stan::math::to_vector;
  using stan::math::value_of;
  using stan::math::var;

  constexpr int dim_theta = 2;
  Eigen::VectorXd theta_0(dim_theta);
  theta_0 << 0, 0;
  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "[          ] [  INFO  ]"
                    << " Skipping test for hessian of size " << theta_0.size()
                    << " with hessian block size of " << hessian_block_size << std::endl;
  }
  JLOG().init_builder("test", "laplace_marginal_poisson_lpmf_" + std::to_string(solver_num) + "_" +
                                  std::to_string(hessian_block_size) + "_" +
                                  std::to_string(max_steps_line_search));

  constexpr double alpha_dbl = 1.6;
  constexpr double rho_dbl = 0.45;
  std::vector<Eigen::VectorXd> x(dim_theta);
  Eigen::VectorXd x_0(2);
  x_0 << 0.05100797, 0.16086164;
  Eigen::VectorXd x_1(2);
  x_1 << -0.59823393, 0.98701425;
  x[0] = x_0;
  x[1] = x_1;

  std::vector<double> delta;
  std::vector<int> delta_int;

  std::vector<int> y = {1, 0};
  std::vector<int> y_index = {1, 2};

  stan::math::test::squared_kernel_functor sq_kernel;
  constexpr double tolerance = 1e-12;
  constexpr int max_num_steps = 100;

  stan::test::ad_tolerances tols;
  // tols.gradient_val_ = 1e-3;
  //  tols.gradient_grad_ = 1e-3;
  auto f = [&](auto&& alpha, auto&& rho) {
    auto __b = JLOG().builder();
    __b.field("component", "laplace_marginal_poisson_lpmf")
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
      auto val = laplace_marginal_tol_poisson_log_lpmf(
          y, y_index, 0, sq_kernel, std::forward_as_tuple(x, alpha, rho),
          theta_0, tolerance, max_num_steps, hessian_block_size, solver_num,
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
                        "laplace_marginal_poisson_lpmf", __b);
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
                        "laplace_marginal_poisson_lpmf", __b);
      throw e;
    }

  };
  stan::test::expect_ad<true>(tols, f, alpha_dbl, rho_dbl);
}
/*
TEST_P(LaplaceSolverGrid, laplace_marginal_poisson_log_lpmf_log_phi_dim_2_mean) {
  using stan::math::laplace_marginal_poisson_log_lpmf;
  using stan::math::laplace_marginal_tol_poisson_log_lpmf;

  using stan::math::log;
  using stan::math::to_vector;
  using stan::math::value_of;
  using stan::math::var;

  constexpr int dim_theta = 2;
  Eigen::VectorXd theta_0(dim_theta);
  theta_0 << 0, 0;
  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "[          ] [  INFO  ]"
                    << " Skipping test for hessian of size " << theta_0.size()
                    << " with hessian block size of " << hessian_block_size << std::endl;
  }

  constexpr double alpha_dbl = 1.6;
  constexpr double rho_dbl = 0.45;
  std::vector<Eigen::VectorXd> x(dim_theta);
  Eigen::VectorXd x_0(2);
  x_0 << 0.05100797, 0.16086164;
  Eigen::VectorXd x_1(2);
  x_1 << -0.59823393, 0.98701425;
  x[0] = x_0;
  x[1] = x_1;

  std::vector<double> delta;
  std::vector<int> delta_int;

  std::vector<int> y = {1, 0};
  std::vector<int> y_index = {1, 2};

  stan::math::test::squared_kernel_functor sq_kernel;
  constexpr double tolerance = 1e-12;
  constexpr int max_num_steps = 100;

  //  stan::test::ad_tolerances tols;
  // tols.gradient_val_ = 1e-3;
  constexpr stan::test::ad_tolerances tols{
      stan::test::ad_gradient_tols{1e-8, 1e-3}};

  //  tols.gradient_grad_ = 1e-3;
  Eigen::VectorXd ye(2);
  ye << 1, 1;
  auto f = [&](auto&& alpha, auto&& rho) {
    return laplace_marginal_tol_poisson_log_lpmf(
        y, y_index, log(ye), sq_kernel,
        std::forward_as_tuple(x, alpha, rho), theta_0, tolerance,
        max_num_steps, hessian_block_size, solver_num,
        max_steps_line_search, nullptr);
  };
  stan::test::expect_ad<true>(tols, f, alpha_dbl, rho_dbl);
}
*/

INSTANTIATE_TEST_SUITE_P(
    PoissonLogLpmf,
    LaplaceSolverGrid,
    ::testing::Combine(
        ::testing::Values(1, 2, 3),      // solver_num
        ::testing::Values(1, 2, 3),      // hessian_block_size
        ::testing::Values(0, 250)       // max_steps_line_search
    ),
    ParamName);

TEST_P(laplace_disease_map_test, laplace_marginal_poisson_log_lpmf) {
  using stan::math::laplace_marginal_poisson_log_lpmf;
  using stan::math::laplace_marginal_tol_poisson_log_lpmf;
  using stan::math::log;
  using stan::math::value_of;
  using stan::math::var;
  static std::once_flag log_once;
  std::call_once(log_once, [] {
    JLOG().set_file("../disease_map_laplace_marginal_poisson_log_lpmf.jsonl", false);
  });
  static std::atomic<int> run_counter{0};

  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "[          ] [  INFO  ]"
                    << " Skipping test for hessian of size " << theta_0.size()
                    << " with hessian block size of " << hessian_block_size << std::endl;
  }
  JLOG().init_builder("test", "disease_map_laplace_marginal_poisson_log_" + std::to_string(solver_num) + "_" +
                                  std::to_string(hessian_block_size) + "_" +
                                  std::to_string(max_steps_line_search));
  constexpr double tolerance = 1e-12;
  constexpr int max_num_steps = 100;
  /*
  {
    double marginal_density = laplace_marginal_tol_poisson_log_lpmf(
        y, y_index, log(ye), stan::math::test::sqr_exp_kernel_functor(),
        std::forward_as_tuple(x, phi_dbl(0), phi_dbl(1)), theta_0, tolerance,
          max_num_steps, hessian_block_size, solver_num,
          max_steps_line_search, nullptr);

    double tol = 6e-4;
    // Benchmark from GPStuff.
    EXPECT_NEAR(-2866.88, marginal_density, tol);
  }
    */
  auto f = [&](auto&& alpha, auto&& rho) {
    auto __b = JLOG().builder();
    __b.field("component", "disease_map_laplace_marginal_poisson_log")
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
    auto lp_val = laplace_marginal_tol_poisson_log_lpmf(
        y, y_index, log(ye), stan::math::test::sqr_exp_kernel_functor(),
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
                        "disease_map_laplace_marginal_poisson_log", __b);
      return lp_val;

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
                        "disease_map_laplace_marginal_poisson_log", __b);
      throw e;
    }

  };
  stan::test::expect_ad<true>(f, phi_dbl[0], phi_dbl[1]);
}

INSTANTIATE_TEST_SUITE_P(
    PoissonLogLpmf,
    laplace_disease_map_test,
    ::testing::Combine(
        ::testing::Values(1, 2, 3),      // solver_num
        ::testing::Values(1, 2, 3),      // hessian_block_size
        ::testing::Values(0, 250)       // max_steps_line_search
    ),
    ParamName);

struct diag_covariance {
  template <typename T0__>
  Eigen::Matrix<stan::return_type_t<T0__>, -1, -1> operator()(
      const T0__& sigma, const int& N, std::ostream* pstream__) const {
    return stan::math::diag_matrix(
        stan::math::rep_vector(stan::math::pow(sigma, 2), N));
  }
};
/*
TEST(laplace_marginal_poisson_log_lpmf, mean_argument) {
  // working example from
  // https://discourse.mc-stan.org/t/embedded-laplace-numerical-problem/39700
  using stan::math::laplace_marginal_poisson_log_lpmf;

  const int N = 1;
  const std::vector<int> y{153};
  const std::vector<int> y_index{1};

  Eigen::VectorXd mu(1);
  mu << 4.3;

  const double sigmaz = 2.0;

  double marginal_density = laplace_marginal_poisson_log_lpmf(
      y, y_index, mu, diag_covariance(), std::tuple<double, int>(sigmaz, N),
      nullptr);

  EXPECT_FLOAT_EQ(-6.7098737, marginal_density);
}
*/
