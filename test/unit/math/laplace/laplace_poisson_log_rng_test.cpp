#include <stan/math.hpp>
#include <stan/math/mix.hpp>
#include <test/unit/math/laplace/laplace_utility.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

#include <gtest/gtest.h>
class laplace_poisson_count_two_dim_diag_test : public laplace_count_two_dim_diag_test {
  public:
  void SetUp() override {
    using stan::math::algebra_solver;
    theta_root = algebra_solver(stan::math::test::stationary_point(), theta_0,
                                phi, d0, di0);
    K_laplace = stan::math::test::laplace_covariance(theta_root, phi);

    rng.seed(1954);
    theta_benchmark = stan::math::multi_normal_rng(theta_root, K_laplace, rng);
  }
};

TEST_P(laplace_poisson_count_two_dim_diag_test, poisson_log_likelihood_rng) {
  using stan::math::laplace_latent_poisson_log_rng;
  using stan::math::laplace_latent_tol_poisson_log_rng;
  using stan::math::multi_normal_rng;
  using stan::math::sqrt;
  using stan::math::square;
  static std::once_flag log_once;
  std::call_once(log_once, [] {
    JLOG().set_file("../laplace_poisson_two_dim_rng.jsonl", false);
  });

  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "[          ] [  INFO  ]"
                    << " Skipping test for hessian of size " << theta_0.size()
                    << " with hessian block size of " << hessian_block_size << std::endl;
  }
  JLOG().init_builder("test", "laplace_poisson_two_dim_rng_" + std::to_string(solver_num) + "_" +
                                  std::to_string(hessian_block_size) + "_" +
                                  std::to_string(max_steps_line_search));

  rng.seed(1954);
  static std::atomic<int> run_counter{0};
  constexpr double tolerance = 1e-12;
  constexpr int max_num_steps = 100;
  auto __b = JLOG().builder();
  __b.field("component", "laplace_poisson_two_dim_rng")
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
  Eigen::MatrixXd theta_pred = laplace_latent_tol_poisson_log_rng(
      y, y_index, 0, stan::math::test::diagonal_kernel_functor{},
      std::forward_as_tuple(phi(0), phi(1)), theta_0, tolerance,
        max_num_steps, hessian_block_size, solver_num,
        max_steps_line_search, rng, nullptr);
  auto end_t0 = std::chrono::high_resolution_clock::now();
  auto __ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  end_t0 - __t0)
                  .count();
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
                    "laplace_poisson_two_dim_rng", __b);

}
/*
TEST_P(laplace_poisson_count_two_dim_diag_test, poisson_log_likelihood_rng_sim) {
  using stan::math::laplace_latent_poisson_log_rng;
  using stan::math::laplace_latent_tol_poisson_log_rng;
  using stan::math::multi_normal_rng;
  using stan::math::sqrt;
  using stan::math::square;
  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "[          ] [  INFO  ]"
                    << " Skipping test for hessian of size " << theta_0.size()
                    << " with hessian block size of " << hessian_block_size << std::endl;
  }

  rng.seed(1954);
  constexpr double tolerance = 1e-12;
  constexpr int max_num_steps = 100;

  // int n_sim = 5e5;
  Eigen::VectorXd theta_dim0(n_sim);
  Eigen::VectorXd theta_dim1(n_sim);
  for (int i = 0; i < n_sim; i++) {
    rng.seed(2025 + i);
    Eigen::MatrixXd theta_pred = laplace_latent_tol_poisson_log_rng(
        y, y_index, 0, stan::math::test::diagonal_kernel_functor{},
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
  EXPECT_NEAR(K_laplace(0, 0), K_sample(0, 0), 5e-3);
  EXPECT_NEAR(K_laplace(1, 1), K_sample(1, 1), 6e-3);
  EXPECT_NEAR(K_laplace(0, 1), K_sample(0, 1), 1e-3);
}
*/
TEST_P(laplace_poisson_count_two_dim_diag_test, poisson_log_exp_likelihood_rng) {
  using stan::math::laplace_latent_tol_poisson_log_rng;
  using stan::math::log;
  using stan::math::multi_normal_rng;
  using stan::math::sqrt;
  using stan::math::square;
  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "[          ] [  INFO  ]"
                    << " Skipping test for hessian of size " << theta_0.size()
                    << " with hessian block size of " << hessian_block_size << std::endl;
  }
  JLOG().init_builder("test", "laplace_poisson_log_exp_two_dim_rng_" + std::to_string(solver_num) + "_" +
                                  std::to_string(hessian_block_size) + "_" +
                                  std::to_string(max_steps_line_search));

  rng.seed(1954);
  static std::atomic<int> run_counter{0};
  constexpr double tolerance = 1e-12;
  constexpr int max_num_steps = 100;
    auto __b = JLOG().builder();
  __b.field("component", "laplace_poisson_log_exp_two_dim_rng")
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
      .field("phi_01_v", false)
      .field("phi_rest_v", false)
    .end();
  auto __t0 = std::chrono::high_resolution_clock::now();

  Eigen::MatrixXd theta_pred_exp = laplace_latent_tol_poisson_log_rng(
      y, y_index, log(ye), stan::math::test::diagonal_kernel_functor{},
      std::forward_as_tuple(phi(0), phi(1)), theta_0, tolerance,
      max_num_steps, hessian_block_size, solver_num,
      max_steps_line_search, rng, nullptr);
  auto end_t0 = std::chrono::high_resolution_clock::now();
  auto __ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  end_t0 - __t0)
                  .count();
  double tol = 1e-3;
  EXPECT_NEAR(theta_benchmark(0), theta_pred_exp(0), tol);
  EXPECT_NEAR(theta_benchmark(1), theta_pred_exp(1), tol);
  __b.field("v_ns", (long long)__ns);
  if (::testing::Test::HasNonfatalFailure()) {
    __b.field("status","FAILURE");
  } else {
    __b.field("status","SUCCESS");
  }
  JLOG().commit_now(JsonLogger::Level::Debug,
                    "laplace_poisson_log_exp_two_dim_rng", __b);
}
// Not running sim because they take a long time
/*
TEST_P(laplace_poisson_count_two_dim_diag_test, poisson_log_exp_likelihood_rng_sim) {
  using stan::math::laplace_latent_tol_poisson_log_rng;
  using stan::math::log;
  using stan::math::multi_normal_rng;
  using stan::math::sqrt;
  using stan::math::square;
  const auto [solver_num, hessian_block_size, max_steps_line_search] = GetParam();
  if (theta_0.size() % hessian_block_size != 0) {
    GTEST_SKIP() << "[          ] [  INFO  ]"
                    << " Skipping test for hessian of size " << theta_0.size()
                    << " with hessian block size of " << hessian_block_size << std::endl;
  }

  rng.seed(1954);
  constexpr double tolerance = 1e-12;
  constexpr int max_num_steps = 100;

  Eigen::VectorXd theta_dim0(n_sim);
  Eigen::VectorXd theta_dim1(n_sim);
  for (int i = 0; i < n_sim; i++) {
    rng.seed(2025 + i);
    Eigen::MatrixXd theta_pred = laplace_latent_tol_poisson_log_rng(
        y, y_index, log(ye), stan::math::test::diagonal_kernel_functor{},
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
  EXPECT_NEAR(K_laplace(0, 0), K_sample(0, 0), 5e-3);
  EXPECT_NEAR(K_laplace(1, 1), K_sample(1, 1), 6e-3);
  EXPECT_NEAR(K_laplace(0, 1), K_sample(0, 1), 1e-3);
}
*/
INSTANTIATE_TEST_SUITE_P(
    PoissonLogRng,
    laplace_poisson_count_two_dim_diag_test,
    ::testing::Combine(
        ::testing::Values(1, 2, 3),      // solver_num
        ::testing::Values(1, 2, 3),      // hessian_block_size
        ::testing::Values(0, 250)       // max_steps_line_search
    ),
    ParamName);
