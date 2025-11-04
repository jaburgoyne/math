#include <stan/math.hpp>
#include <stan/math/mix.hpp>
#include <test/unit/math/laplace/laplace_utility.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

namespace {
struct stationary_point_nb {
  template <typename T0, typename T1>
  inline Eigen::Matrix<typename stan::return_type<T0, T1>::type, Eigen::Dynamic,
                       1>
  operator()(const Eigen::Matrix<T0, Eigen::Dynamic, 1>& theta,
             const Eigen::Matrix<T1, Eigen::Dynamic, 1>& parms,
             const std::vector<double>& dat, const std::vector<int>& dat_int,
             std::ostream* pstream__ = 0) const {
    using stan::math::exp;
    Eigen::Matrix<typename stan::return_type<T0, T1>::type, Eigen::Dynamic, 1>
        z(2);
    double eta = dat[0];
    Eigen::Matrix<T0, Eigen::Dynamic, 1> exp_theta = exp(theta);
    std::vector<int> y = {1, 0};

    z(0) = -exp_theta(0) * (y[0] + eta) / (eta + exp_theta(0)) + y[0]
           - theta(0) / parms(0);
    z(1) = -exp_theta(1) * (y[1] + eta) / (eta + exp_theta(1)) + y[1]
           - theta(1) / parms(1);
    // z(0) = 1 - (1 - eta) / (1 + eta * exp(theta(0))) - theta(0) / parms(0);
    // z(1) = 0 - (0 - eta) / (1 + eta * exp(theta(1))) - theta(1) / parms(1);
    return z;
  }
};

struct diagonal_kernel_nb_functor {
  template <typename T1, typename T2>
  auto operator()(const T1& alpha, const T2& rho,
                  std::ostream* msgs = nullptr) const {
    Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> K(2, 2);
    K(0, 0) = alpha;
    K(1, 1) = rho;
    K(0, 1) = 0;
    K(1, 0) = 0;
    return K;
  }
};

template <typename T1, typename T2>
Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> laplace_covariance_nb(
    const Eigen::Matrix<T1, Eigen::Dynamic, 1>& theta,
    const Eigen::Matrix<T2, Eigen::Dynamic, 1>& phi, const double& eta) {
  using stan::math::exp;
  using stan::math::square;
  Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> K(2, 2);
  Eigen::Matrix<T1, Eigen::Dynamic, 1> exp_theta = exp(theta);
  std::vector<int> y = {1, 0};
  K(0, 0) = 1
            / ((eta * exp_theta(0) * (y[0] + eta) / square(eta + exp_theta(0)))
               + 1 / phi(0));
  K(1, 1) = 1
            / ((eta * exp_theta(1) * (y[1] + eta) / square(eta + exp_theta(1)))
               + 1 / phi(1));

  // K(0, 0) = 1 / (1 / phi(0) + (1 - eta) * exp(theta(0))
  //                                          / square(1 + eta *
  //                                          exp(theta(0))));
  // K(1, 1) = 1 / (1 / phi(1) + (0 - eta) * exp(theta(1))
  //                                          / square(1 + eta *
  //                                          exp(theta(1))));
  K(0, 1) = 0;
  K(1, 0) = 0;
  return K;
}

class laplace_neg_binomial_count_two_dim_diag_test : public laplace_count_two_dim_diag_test {
  public:
  void SetUp() override {
    using stan::math::algebra_solver;
    d0 = {eta};
    std::vector<int> di0;
    using stan::math::multi_normal_rng;

    theta_root
        = algebra_solver(stationary_point_nb(), theta_0, phi, d0, di0);
    K_laplace = laplace_covariance_nb(theta_root, phi, eta);

    rng.seed(1954);
   theta_benchmark
      = stan::math::multi_normal_rng(theta_root, K_laplace, rng);
  }
};

TEST_P(laplace_neg_binomial_count_two_dim_diag_test, laplace_latent_neg_binomial_2_log_rng) {
  using stan::math::algebra_solver;
  using stan::math::laplace_latent_neg_binomial_2_log_rng;
  using stan::math::laplace_latent_tol_neg_binomial_2_log_rng;
  using stan::math::multi_normal_rng;
  using stan::math::sqrt;
  using stan::math::square;
  static std::once_flag log_once;
  std::call_once(log_once, [] {
    JLOG().set_file("../laplace_neg_binomial_2_log_rng.jsonl", false);
  });

  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "[          ] [  INFO  ]"
                    << " Skipping test for hessian of size " << theta_0.size()
                    << " with hessian block size of " << hessian_block_size << std::endl;
  }
  constexpr double tolerance = 1e-12;
  constexpr int max_num_steps = 100;
  rng.seed(1954);
  static std::atomic<int> run_counter{0};
  JLOG().init_builder("test", "laplace_neg_binomial_2_log_rng_" + std::to_string(solver_num) + "_" +
                                  std::to_string(hessian_block_size) + "_" +
                                  std::to_string(max_steps_line_search));
    auto __b = JLOG().builder();
    __b.field("component", "laplace_neg_binomial_2_log_rng")
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
  Eigen::MatrixXd theta_pred = laplace_latent_tol_neg_binomial_2_log_rng(
      y, y_index, eta, 0, diagonal_kernel_nb_functor{},
      std::forward_as_tuple(phi(0), phi(1)), theta_0, tolerance,
        max_num_steps, hessian_block_size, solver_num,
        max_steps_line_search, rng, nullptr);
    auto end_t0 = std::chrono::high_resolution_clock::now();
    auto __ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end_t0 - __t0)
                    .count();
    __b.field("v_ns", (long long)__ns);
  double tol = 1e-3;
  EXPECT_NEAR(theta_benchmark(0), theta_pred(0), tol);
  EXPECT_NEAR(theta_benchmark(1), theta_pred(1), tol);
    if (::testing::Test::HasNonfatalFailure()) {
      __b.field("status","FAILURE");
    } else {
      __b.field("status","SUCCESS");
    }
    JLOG().commit_now(JsonLogger::Level::Debug,
                      "laplace_neg_binomial_2_log_rng", __b);

}
/*
TEST_P(laplace_neg_binomial_count_two_dim_diag_test, laplace_latent_neg_binomial_2_log_rng_sim) {
  using stan::math::algebra_solver;
  using stan::math::laplace_latent_neg_binomial_2_log_rng;
  using stan::math::laplace_latent_tol_neg_binomial_2_log_rng;
  using stan::math::multi_normal_rng;
  using stan::math::sqrt;
  using stan::math::square;
  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "[          ] [  INFO  ]"
                    << " Skipping test for hessian of size " << theta_0.size()
                    << " with hessian block size of " << hessian_block_size << std::endl;
  }
  constexpr double tolerance = 1e-12;
  constexpr int max_num_steps = 100;
  Eigen::VectorXd theta_dim0(n_sim);
  Eigen::VectorXd theta_dim1(n_sim);
  for (int i = 0; i < n_sim; i++) {
    rng.seed(2025 + i);
    Eigen::MatrixXd theta_pred = laplace_latent_tol_neg_binomial_2_log_rng(
        y, y_index, eta, 0, diagonal_kernel_nb_functor{},
        std::forward_as_tuple(phi(0), phi(1)), theta_0, tolerance,
        max_num_steps, hessian_block_size, solver_num,
        max_steps_line_search, rng, nullptr);

    theta_dim0(i) = theta_pred(0);
    theta_dim1(i) = theta_pred(1);
  }

  Eigen::MatrixXd K_sample(2, 2);
  K_sample(0, 0)
      = theta_dim0.array().square().mean() - square(theta_dim0.mean());
  K_sample(1, 1)
      = theta_dim1.array().square().mean() - square(theta_dim1.mean());
  K_sample(0, 1) = theta_dim0.cwiseProduct(theta_dim1).mean()
                   - theta_dim0.mean() * theta_dim1.mean();
  K_sample(1, 0) = K_sample(0, 1);

  // Check answers are within three std of the true answer.
  EXPECT_NEAR(theta_root(0), theta_dim0.mean(),
              3 * sqrt(K_laplace(0, 0) / n_sim));
  EXPECT_NEAR(theta_root(1), theta_dim1.mean(),
              3 * sqrt(K_laplace(1, 1) / n_sim));

  // Check sample covariance
  EXPECT_NEAR(K_laplace(0, 0), K_sample(0, 0), 6e-3);
  EXPECT_NEAR(K_laplace(1, 1), K_sample(1, 1), 6e-3);
  EXPECT_NEAR(K_laplace(0, 1), K_sample(0, 1), 1e-3);
}
*/
INSTANTIATE_TEST_SUITE_P(
    NegBinomialRng,
    laplace_neg_binomial_count_two_dim_diag_test,
    ::testing::Combine(
        ::testing::Values(1, 2, 3),      // solver_num
        ::testing::Values(1, 2, 3),      // hessian_block_size
        ::testing::Values(0, 250)       // max_steps_line_search
    ),
    ParamName);

}  // namespace
