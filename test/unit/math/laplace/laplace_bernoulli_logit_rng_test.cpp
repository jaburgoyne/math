#include <stan/math.hpp>
#include <stan/math/mix.hpp>

#include <test/unit/math/laplace/laplace_utility.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

namespace {
struct stationary_point {
  template <typename T0, typename T1>
  inline Eigen::Matrix<typename stan::return_type<T0, T1>::type, Eigen::Dynamic,
                       1>
  operator()(const Eigen::Matrix<T0, Eigen::Dynamic, 1>& theta,
             const Eigen::Matrix<T1, Eigen::Dynamic, 1>& parms,
             const std::vector<double>& dat, const std::vector<int>& dat_int,
             std::ostream* pstream__ = 0) const {
    Eigen::Matrix<typename stan::return_type<T0, T1>::type, Eigen::Dynamic, 1>
        z(2);
    z(0) = 1 / (1 + exp(theta(0))) - theta(0) / (parms(0) * parms(0));
    z(1) = -1 / (1 + exp(-theta(1))) - theta(1) / (parms(1) * parms(1));
    return z;
  }
};

struct diagonal_kernel_functor {
  template <typename T1, typename T2>
  auto operator()(const T1& arg1, const T2& arg2,
                  std::ostream* msgs = nullptr) const {
    Eigen::Matrix<stan::return_type_t<T1, T2>, Eigen::Dynamic, Eigen::Dynamic>
        K(2, 2);
    K(0, 0) = arg1 * arg1;
    K(1, 1) = arg2 * arg2;
    K(0, 1) = 0;
    K(1, 0) = 0;
    return K;
  }
};

template <typename T1, typename T2>
Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> laplace_covariance(
    const Eigen::Matrix<T1, Eigen::Dynamic, 1>& theta,
    const Eigen::Matrix<T2, Eigen::Dynamic, 1>& phi) {
  using stan::math::exp;
  using stan::math::square;
  Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> K(2, 2);
  K(0, 0)
      = -1
        / (-1 / (phi(0) * phi(0)) - exp(theta(0)) / square(1 + exp(theta(0))));
  K(1, 1) = -1
            / (-1 / (phi(1) * phi(1))
               - exp(-theta(1)) / square(1 + exp(-theta(1))));
  K(0, 1) = 0;
  K(1, 0) = 0;
  return K;
}

class laplace_bernoulli_logit_rng : public ::testing::TestWithParam<std::tuple<int,int,int>> {};

TEST_P(laplace_bernoulli_logit_rng, two_dim_diag) {
  static std::once_flag log_once;
  std::call_once(log_once, [] {
    JLOG().set_file("../laplace_bernoulli_two_dim_rng.jsonl", false);
  });

  using stan::math::algebra_solver;
  using stan::math::laplace_latent_tol_bernoulli_logit_rng;
  using stan::math::multi_normal_rng;
  using stan::math::sqrt;
  using stan::math::square;
  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();

  Eigen::VectorXd theta_0{{0, 0}};
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "[          ] [  INFO  ]"
                    << " Skipping test for hessian of size " << theta_0.size()
                    << " with hessian block size of " << hessian_block_size << std::endl;
  }
  JLOG().init_builder("test", "laplace_bernoulli_two_dim_rng_" + std::to_string(solver_num) + "_" +
                                  std::to_string(hessian_block_size) + "_" +
                                  std::to_string(max_steps_line_search));
  Eigen::VectorXd phi{{3, 2}};
  std::vector<int> n_samples = {1, 1};
  std::vector<int> sums = {1, 0};
  Eigen::VectorXd ye{{1, 1}};
  Eigen::VectorXd mean{{0, 0}};
  std::vector<double> d0;
  std::vector<int> di0;
  boost::random::mt19937 rng;
  rng.seed(1954);
  static std::atomic<int> run_counter{0};
  constexpr double tolerance = 1e-12;
  constexpr int64_t max_num_steps = 1e4;
  auto __b = JLOG().builder();
  __b.field("component", "laplace_bernoulli_two_dim_rng")
    .field("where", "param_test")
    .field("event", "laplace_marginal_tol_call")
    .field("v_level", 0)
    .field("run_num", ++run_counter)
    .begin_object("test")
      .field("solver_num", solver_num)
      .field("hessian_block_size", hessian_block_size)
      .field("max_steps_line_search", max_steps_line_search)
    .end();
  auto __t0 = std::chrono::high_resolution_clock::now();
  Eigen::MatrixXd theta_pred = laplace_latent_tol_bernoulli_logit_rng(
      sums, n_samples, mean, diagonal_kernel_functor{},
      std::forward_as_tuple(phi(0), phi(1)), theta_0, tolerance,
        max_num_steps, hessian_block_size, solver_num,
        max_steps_line_search, rng, nullptr);
  auto end_t0 = std::chrono::high_resolution_clock::now();
  auto __ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  end_t0 - __t0)
                  .count();

  // Compute exact mean and covariance
  Eigen::VectorXd theta_root
      = algebra_solver(stationary_point{}, theta_0, phi, d0, di0);
  Eigen::MatrixXd K_laplace = laplace_covariance(theta_root, phi);

  rng.seed(1954);
  Eigen::MatrixXd theta_benchmark
      = multi_normal_rng(theta_root, K_laplace, rng);

  double tol = 1e-3;
  EXPECT_NEAR(theta_benchmark(0), theta_pred(0), tol);
  EXPECT_NEAR(theta_benchmark(1), theta_pred(1), tol);
  __b.field("v_ns", (long long)__ns);
  if (::testing::Test::HasNonfatalFailure()) {
    __b.field("status","FAILURE");
  } else {
    __b.field("status","SUCCESS");
  }
  JLOG().commit_now(JsonLogger::Level::Debug,
                    "laplace_bernoulli_two_dim_rng", __b);
}

INSTANTIATE_TEST_SUITE_P(
    BernoulliLogitRngMapParamTestSuite,
    laplace_bernoulli_logit_rng,
    ::testing::Combine(
        ::testing::Values(1, 2, 3),      // solver_num
        ::testing::Values(1, 2, 3),      // hessian_block_size
        ::testing::Values(0, 250)       // max_steps_line_search
    ),
    ParamName);


}  // namespace
