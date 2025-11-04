#include <test/unit/math/test_ad.hpp>
#include <stan/math.hpp>
#include <stan/math/mix.hpp>
#include <test/unit/math/laplace/aki_synth_data/x1.hpp>
#include <test/unit/math/laplace/laplace_utility.hpp>
#include <test/unit/math/rev/fun/util.hpp>
#include <stan/math/prim/fun/lgamma.hpp>
#include <test/unit/math/laplace/motorcycle_gp/x_vec.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

struct poisson_log_likelihood2 {
  template <typename Theta>
  auto operator()(const Theta& theta, const std::vector<int>& delta_int,
                  std::ostream* pstream) const {
    return stan::math::poisson_log_lpmf(delta_int, theta);
  }
};


class PoissonLogPhiDim2
    : public ::testing::TestWithParam<std::tuple<int,int,int>>  { // solver, hblock, ls_steps
 protected:
  // Optionally put per-test setup here (we'll re-init theta0 inside TEST_P).
  void SetUp() override {}
 // logger->current_test_name_ = "poisson_log_phi_dim_2";
  static constexpr int dim_phi = 2;
  Eigen::Matrix<double, Eigen::Dynamic, 1> phi_dbl{{1.6, 0.45}};

  static constexpr int dim_theta = 2;
  Eigen::VectorXd theta_0{{0.0, 0.0}};
  std::vector<Eigen::VectorXd> x{Eigen::VectorXd{{0.05100797, 0.16086164}}, Eigen::VectorXd{{-0.59823393, 0.98701425}}};
  Eigen::VectorXd y_dummy;
  std::vector<int> n_samples{1, 1};
  std::vector<int> sums{1, 0};
};

static std::once_flag log_once;

TEST_P(PoissonLogPhiDim2, poisson_log_phi_dim_2) {
  using stan::math::laplace_marginal_tol;

  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();

  // One-time log sink initialization (safe across test cases)
  std::call_once(log_once, [] {
    JLOG().set_file("../laplace_lpdf.jsonl", false);
  });
  JLOG().init_builder("test", "poisson_log_phi_dim_2_" + std::to_string(solver_num) + "_" +
                                  std::to_string(hessian_block_size) + "_" +
                                  std::to_string(max_steps_line_search));

  using stan::math::laplace_marginal;
  using stan::math::laplace_marginal_tol;
  using stan::math::to_vector;
  using stan::math::value_of;
  using stan::math::var;


  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "[          ] [  INFO  ]"
                    << " Skipping test for hessian of size " << theta_0.size()
                    << " with hessian block size of " << hessian_block_size << std::endl;
  }
/*
  double target = laplace_marginal<false>(
      poisson_log_likelihood2{}, std::forward_as_tuple(sums),
      stan::math::test::squared_kernel_functor{},
      std::forward_as_tuple(x, phi_dbl(0), phi_dbl(1)), nullptr);

  // TODO(Charles): benchmark target against gpstuff.
  constexpr double tol = 1e-4;
  EXPECT_NEAR(-2.53056, value_of(target), tol);
  // Test with optional arguments
  {
    constexpr double tolerance = 1e-12;
    constexpr int max_num_steps = 100;
    constexpr int hessian_block_size = 1;
    constexpr int solver = 1;
    constexpr int max_steps_line_search = 10;

    target = laplace_marginal_tol<false>(
        poisson_log_likelihood2{}, std::forward_as_tuple(sums),
        stan::math::test::squared_kernel_functor{},
        std::forward_as_tuple(x, phi_dbl(0), phi_dbl(1)), theta_0, tolerance,
        max_num_steps, hessian_block_size, solver, max_steps_line_search,
        nullptr);
    EXPECT_NEAR(-2.53056, value_of(target), tol);
  }
*/

  constexpr double tolerance = 1e-12;
  constexpr int max_num_steps = 100;
  using stan::is_var_v;
  using stan::scalar_type_t;
  constexpr stan::test::ad_tolerances tols{
      stan::test::ad_gradient_tols{1e-8, 1e-3}};
  //  tols.gradient_grad_ = 1e-3;
  int run_num = 0;
        auto f = [&](auto&& x_v, auto&& alpha, auto&& rho) {
          JLOG().init_builder("test", "poisson_log_phi_dim_2" + std::to_string(solver_num) + "_" +
                                          std::to_string(hessian_block_size) + "_" +
                                          std::to_string(max_steps_line_search));
          auto __b = JLOG().builder();
          __b.field("component","poisson_log_phi_dim_2")
            .field("where","run_solver_grid")
            .field("event","laplace_marginal_tol_call")
            .field("v_level", 0)
            .field("run_num", ++run_num)
            .begin_object("test")
              .field("solver_num", solver_num)
              .field("hessian_block_size", hessian_block_size)
              .field("max_steps_line_search", max_steps_line_search)
            .end()
            .begin_object("autodiff")
              .field("x_v", (bool)stan::is_any_autodiff_v<decltype(x_v)>)
              .field("alpha", (bool)stan::is_any_autodiff_v<decltype(alpha)>)
              .field("rho", (bool)stan::is_any_autodiff_v<decltype(rho)>)
            .end();
          auto __t0 = std::chrono::high_resolution_clock::now();
          try {
            auto lp_val = laplace_marginal_tol<false>(
                poisson_log_likelihood2{}, std::forward_as_tuple(sums),
                stan::math::test::squared_kernel_functor{},
                std::forward_as_tuple(x_v, alpha, rho), theta_0, tolerance,
                max_num_steps, hessian_block_size, solver_num,
                max_steps_line_search, nullptr);
            auto end_t0 = std::chrono::high_resolution_clock::now();
            auto __ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_t0 - __t0).count();
            __b.field("v_ns",(long long)__ns);
            if (::testing::Test::HasNonfatalFailure()) {
              __b.field("status","FAILURE");
            } else {
              __b.field("status","SUCCESS");
            }
            JLOG().commit_now(JsonLogger::Level::Debug, "poisson_log_phi_dim_2", __b);
          return lp_val;
          } catch (const std::exception& e) {
            auto end_t0 = std::chrono::high_resolution_clock::now();
            auto __ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_t0 - __t0).count();
            __b.field("error", e.what());
            __b.field("v_ns",(long long)__ns);
            if (::testing::Test::HasNonfatalFailure()) {
              __b.field("status","FAILURE");
            } else {
              __b.field("status","SUCCESS");
            }
            JLOG().commit_now(JsonLogger::Level::Debug, "poisson_log_phi_dim_2", __b);
            throw e;
          }

        };
        stan::test::expect_ad<true>(tols, f, x, phi_dbl[0], phi_dbl[1]);
}

// Instantiate over the full grid: solver ∈ {1,2,3}, hblock ∈ {1,2,3}, ls_steps ∈ {0,1000}.
INSTANTIATE_TEST_SUITE_P(
    PoissonLogPhiDim2ParamTestSuite,
    PoissonLogPhiDim2,
    ::testing::Combine(
        ::testing::Values(1, 2, 3),      // solver_num
        ::testing::Values(1, 2, 3),      // hessian_block_size
        ::testing::Values(0, 250)       // max_steps_line_search
    ),
    ParamName);

struct poisson_log_exposure_likelihood {
  template <typename Theta, typename YEVec>
  auto operator()(const Theta& theta, YEVec&& ye,
                  const std::vector<int>& delta_int,
                  std::ostream* pstream) const {
    return stan::math::poisson_log_lpmf(
        delta_int, stan::math::add(theta, stan::math::log(ye)));
  }
};
TEST_P(laplace_disease_map_test, laplace_marginal_val) {
  using stan::math::laplace_marginal;
  using stan::math::laplace_marginal_poisson_log_lpmf;
  using stan::math::laplace_marginal_tol;
  using stan::math::value_of;
  using stan::math::var;
  using stan::math::laplace_marginal_tol;

  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "Skipping: theta_0.size() = " << theta_0.size()
                 << " not divisible by hessian_block_size = "
                 << hessian_block_size;
  }
  // One-time log sink initialization (safe across test cases)
  JLOG().init_builder("test", "disease_map_laplace_lpdf_" + std::to_string(solver_num) + "_" +
                                  std::to_string(hessian_block_size) + "_" +
                                  std::to_string(max_steps_line_search));
  constexpr double tolerance = 1e-12;
  constexpr int max_num_steps = 100;
  double marginal_density = laplace_marginal_tol<false>(
      poisson_log_exposure_likelihood{}, std::forward_as_tuple(ye, y),
      stan::math::test::sqr_exp_kernel_functor{},
      std::forward_as_tuple(x, phi_dbl(0), phi_dbl(1)), theta_0, tolerance,
                max_num_steps, hessian_block_size, solver_num,
                max_steps_line_search, nullptr);

  constexpr double tol = 6e-4;
  // Benchmark from GPStuff.
  EXPECT_NEAR(-2866.88, value_of(marginal_density), tol);
}
TEST_P(laplace_disease_map_test, laplace_marginal_lpdf) {
  using stan::math::laplace_marginal;
  using stan::math::laplace_marginal_poisson_log_lpmf;
  using stan::math::laplace_marginal_tol;
  using stan::math::value_of;
  using stan::math::var;
  using stan::math::laplace_marginal_tol;

  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "Skipping: theta_0.size() = " << theta_0.size()
                 << " not divisible by hessian_block_size = "
                 << hessian_block_size;
  }

  // One-time log sink initialization (safe across test cases)
  JLOG().init_builder("test", "disease_map_laplace_lpdf_" + std::to_string(solver_num) + "_" +
                                  std::to_string(hessian_block_size) + "_" +
                                  std::to_string(max_steps_line_search));

  constexpr double tolerance = 1e-12;
  constexpr int max_num_steps = 100;
  int run_num = 0;
        auto f = [&](auto&& alpha, auto&& rho) {
          auto __b = JLOG().builder();
          __b.field("component","poisson_log_phi_dim_2")
            .field("where","run_solver_grid")
            .field("event","laplace_marginal_tol_call")
            .field("v_level", 0)
            .field("run_num", ++run_num)
            .begin_object("test")
              .field("solver_num", solver_num)
              .field("hessian_block_size", hessian_block_size)
              .field("max_steps_line_search", max_steps_line_search)
            .end()
            .begin_object("autodiff")
              .field("alpha", (bool)stan::is_any_autodiff_v<decltype(alpha)>)
              .field("rho", (bool)stan::is_any_autodiff_v<decltype(rho)>)
            .end();
          auto __t0 = std::chrono::high_resolution_clock::now();
          try {
            auto lp_val = laplace_marginal_tol<false>(
                poisson_log_exposure_likelihood{}, std::forward_as_tuple(ye, y),
                stan::math::test::sqr_exp_kernel_functor{},
                std::forward_as_tuple(x, alpha, rho), theta_0, tolerance,
                max_num_steps, hessian_block_size, solver_num,
                max_steps_line_search, nullptr);
            auto end_t0 = std::chrono::high_resolution_clock::now();
            auto __ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_t0 - __t0).count();
            __b.field("v_ns",(long long)__ns);
            if (::testing::Test::HasNonfatalFailure()) {
              __b.field("status","FAILURE");
            } else {
              __b.field("status","SUCCESS");
            }
            JLOG().commit_now(JsonLogger::Level::Debug, "laplace_marginal_disease_map", __b);
            return lp_val;
          } catch (const std::exception& e) {
            auto end_t0 = std::chrono::high_resolution_clock::now();
            auto __ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_t0 - __t0).count();
            __b.field("error", e.what());
            __b.field("v_ns",(long long)__ns);
            if (::testing::Test::HasNonfatalFailure()) {
              __b.field("status","FAILURE");
            } else {
              __b.field("status","SUCCESS");
            }
            JLOG().commit_now(JsonLogger::Level::Debug, "laplace_marginal_disease_map", __b);
            throw e;
          }
        };
        stan::test::expect_ad<true>(f, phi_dbl[0], phi_dbl[1]);
}

INSTANTIATE_TEST_SUITE_P(
    LaplaceDiseaseMapParamTestSuite,
    laplace_disease_map_test,
    ::testing::Combine(
        ::testing::Values(1, 2, 3),      // solver_num
        ::testing::Values(1, 2, 3),      // hessian_block_size
        ::testing::Values(0, 250)       // max_steps_line_search
    ),
    ParamName);


struct bernoulli_logit_likelihood {
  template <typename Theta>
  auto operator()(const Theta& theta, const std::vector<int>& delta_int,
                  std::ostream* pstream) const {
    return stan::math::bernoulli_logit_lpmf(delta_int, theta);
  }
};
TEST_P(bernoulli_logit_phi_dim500, laplace_lpdf_test) {
  using stan::math::laplace_marginal;
  using stan::math::laplace_marginal_tol;
  using stan::math::to_vector;
  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "Skipping: theta_0.size() = " << theta_0.size()
                 << " not divisible by hessian_block_size = "
                 << hessian_block_size;
  }

  constexpr stan::test::ad_tolerances tols{
      stan::test::ad_gradient_tols{1e-8, 5e-3}};
  int run_num = 0;
        auto f = [&](auto&& alpha, auto&& rho) {
        JLOG().init_builder("test", "bernoulli_logit_dim_500_" + std::to_string(solver_num) + "_" +
                                        std::to_string(hessian_block_size) + "_" +
                                        std::to_string(max_steps_line_search));

        auto __b = JLOG().builder();
        __b.field("component","bernoulli_logit_dim_500")
          .field("where","run_solver_grid")
          .field("event","laplace_marginal_tol_call")
          .field("v_level", 0)
          .field("run_num", ++run_num)
          .begin_object("test")
            .field("solver_num", solver_num)
            .field("hessian_block_size", hessian_block_size)
            .field("max_steps_line_search", max_steps_line_search)
          .end()
          .begin_object("autodiff")
            .field("alpha", (bool)stan::is_any_autodiff_v<decltype(alpha)>)
            .field("rho", (bool)stan::is_any_autodiff_v<decltype(rho)>)
          .end();
          auto __t0 = std::chrono::high_resolution_clock::now();
            try {
            auto lp_val = laplace_marginal_tol<false>(
                bernoulli_logit_likelihood{}, std::forward_as_tuple(y),
                stan::math::test::sqr_exp_kernel_functor{},
                std::forward_as_tuple(x, alpha, rho), theta_0, tolerance,
                max_num_steps, hessian_block_size, solver_num,
                max_steps_line_search, nullptr);
            auto end_t0 = std::chrono::high_resolution_clock::now();
            auto __ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_t0 - __t0).count();
            __b.field("v_ns",(long long)__ns);
            if (::testing::Test::HasNonfatalFailure()) {
              __b.field("status","FAILURE");
            } else {
              __b.field("status","SUCCESS");
            }
            JLOG().commit_now(JsonLogger::Level::Debug, "bernoulli_logit_dim_500", __b);
            return lp_val;
            } catch (const std::exception& e) {
            auto end_t0 = std::chrono::high_resolution_clock::now();
            auto __ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_t0 - __t0).count();
            __b.field("error", e.what());
            __b.field("v_ns",(long long)__ns);
            if (::testing::Test::HasNonfatalFailure()) {
              __b.field("status","FAILURE");
            } else {
              __b.field("status","SUCCESS");
            }
            JLOG().commit_now(JsonLogger::Level::Debug, "gp_motorcycle_ad", __b);
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
        ::testing::Values(0, 250)       // max_steps_line_search
    ),
    ParamName);
