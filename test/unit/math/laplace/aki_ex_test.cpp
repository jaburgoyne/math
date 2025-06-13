#include <test/unit/math/test_ad.hpp>
#include <stan/math.hpp>
#include <stan/math/mix.hpp>
#include <test/unit/math/laplace/laplace_utility.hpp>
#include <test/unit/math/laplace/aki_synth_data/x1.hpp>

#include <test/unit/math/rev/fun/util.hpp>

#include <gtest/gtest.h>
#include <iostream>
#include <istream>
#include <fstream>
#include <vector>

// --- Minimal stubs for the Stan functors and math calls ---

// real poisson_re_log_ll(vector, data array[] int, vector)
template <typename T0__, typename T2__>
stan::return_type_t<stan::base_type_t<T0__>, stan::base_type_t<T2__>>
poisson_re_log_ll(const T0__& theta_arg__, const std::vector<int>& y,
                  const T2__& mu_arg__, std::ostream* pstream__) {
  using local_scalar_t__ = stan::return_type_t<stan::base_type_t<T0__>,
                             stan::base_type_t<T2__>>;
  const auto& theta = stan::math::to_ref(theta_arg__);
  const auto& mu = stan::math::to_ref(mu_arg__);
    return stan::math::poisson_log_lpmf<false>(y, stan::math::add(mu, theta));

}

// matrix cov_fun(real, data int)
template <typename T0__>
Eigen::Matrix<stan::return_type_t<T0__>,-1,-1>
cov_fun(const T0__& sigma, const int& N, std::ostream* pstream__) {
  using local_scalar_t__ = stan::return_type_t<T0__>;
    return stan::math::diag_matrix(
             stan::math::rep_vector(stan::math::pow(sigma, 2), N));

}

// real integrand(real, real, array[] real, array[] real, array[] int)
template <typename T0__, typename T1__, typename T2__, typename T3__,
          typename T4__>
stan::return_type_t<T0__, T1__, stan::base_type_t<T2__>,
  stan::base_type_t<T3__>>
integrand(const T0__& theta, const T1__& notused, const T2__& phi,
          const T3__& X_i, const T4__& y_i, std::ostream* pstream__) {
  using local_scalar_t__ = stan::return_type_t<T0__, T1__,
                             stan::base_type_t<T2__>,
                             stan::base_type_t<T3__>>;
  // suppress unused var warning
  static constexpr bool propto__ = true;
  local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    local_scalar_t__ sigma = DUMMY_VAR__;
    current_statement__ = 23;
    sigma = phi(0);
    local_scalar_t__ mu = DUMMY_VAR__;
    current_statement__ = 24;
    mu = phi(1);
    local_scalar_t__ p = DUMMY_VAR__;
    current_statement__ = 25;
    p = stan::math::exp((stan::math::normal_lpdf<false>(theta, 0, sigma) +
          stan::math::poisson_log_lpmf<false>(y_i, (theta + mu))));
    current_statement__ = 26;
    return p;

}

struct poisson_re_log_ll_functor__ {
  template <typename T0__, typename T2__>
  stan::return_type_t<stan::base_type_t<T0__>, stan::base_type_t<T2__>>
  operator()(const T0__& theta, const std::vector<int>& y, const T2__& mu,
             std::ostream* pstream__) const {
    return poisson_re_log_ll(theta, y, mu, pstream__);
  }
};
struct integrand_functor__ {
  template <typename T0__, typename T1__, typename T2__, typename T3__,
            typename T4__>
  stan::return_type_t<T0__, T1__, stan::base_type_t<T2__>,
    stan::base_type_t<T3__>>
  operator()(const T0__& theta, const T1__& notused, const T2__& phi,
             const T3__& X_i, const T4__& y_i, std::ostream* pstream__) const {
    return integrand(theta, notused, phi, X_i, y_i, pstream__);
  }
};
struct cov_fun_functor__ {
  template <typename T0__>
  Eigen::Matrix<stan::return_type_t<T0__>,-1,-1>
  operator()(const T0__& sigma, const int& N, std::ostream* pstream__) const {
    return cov_fun(sigma, N, pstream__);
  }
};

// --- The test itself ---

TEST(WriteArrayBodySimple, ExecutesBodyWithHardcodedData) {
  const int  N = 1;
  const std::vector<int>    y{153};
  const std::vector<double> mu{4.3};
  const double sigmaz = 2.0;
  const double integrate_1d_reltol = 1e-6;
  std::ostream* pstream__ = nullptr;
  auto ll_laplace = stan::math::laplace_marginal(
      poisson_re_log_ll_functor__(),
      std::tuple<const std::vector<int>&, const std::vector<double>&>(y, mu),
      /*theta_0=*/Eigen::VectorXd::Zero(N),
      cov_fun_functor__(),
      std::tuple<double, int>(sigmaz, N),
      pstream__);

  double ll_integrate_1d = 0;
  for (int i = 1; i <= N; ++i) {
    double piece = stan::math::integrate_1d(
      integrand_functor__(),
      stan::math::negative_infinity(),
      stan::math::positive_infinity(),
      stan::math::append_array(
        std::vector<double>{sigmaz},
        std::vector<double>{ mu[i-1] }
      ),
      std::vector<double>{0},
      std::vector<int>{ y[i-1] },
      pstream__,
      integrate_1d_reltol
    );
    ll_integrate_1d += std::log(piece);
  }
  std::cout << "Laplace result: " << ll_laplace << std::endl;
  std::cout << "Integrated result: " << ll_integrate_1d << std::endl;
  // --- end inlined body ---

  // Assertions
  EXPECT_TRUE(std::isfinite(ll_laplace)) << "Laplace result should be finite";
  EXPECT_TRUE(std::isfinite(ll_integrate_1d)) << "Integrated result should be finite";
}
