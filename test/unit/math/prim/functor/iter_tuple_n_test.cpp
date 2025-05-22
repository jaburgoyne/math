#include <stan/math/prim.hpp>
#include <test/unit/util.hpp>
#include <test/unit/math/prim/util.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>
namespace {
TEST(MathPrim, iter_tuple_n_types) {
  using stan::math::iter_tuple_n;
  auto x_tuple = 
  std::make_tuple(1, 1, 
    Eigen::VectorXd{{1, 1, 1}},
    std::make_tuple(1, 
      std::vector{
        std::make_tuple(1, Eigen::VectorXd{{1, 1, 1}})}));
  std::vector<decltype(x_tuple)> x_vec;
  x_vec.push_back(x_tuple);
  x_vec.push_back(x_tuple);
  iter_tuple_n(
      [](auto&& x) {
        using stan::is_eigen_v;
        using x_t = std::decay_t<decltype(x)>;
        constexpr bool is_vec_container = stan::is_std_vector_v<x_t> && !stan::is_stan_scalar<stan::value_type_t<x_t>>::value;
        if constexpr (is_eigen_v<decltype(x)>) {
          EXPECT_EQ(x.size(), 3);
          x.array() += 1.0;
        } else if constexpr (is_vec_container) {
          for (int i = 0; i < x.size(); ++i) {
            x[i] += 1;
          }
        } else {
          x += 1;
        }
      },
      std::forward<decltype(x_vec)>(x_vec));
  for (const auto& x : x_vec) {
    // Check each scalar element is equal to 2
    EXPECT_EQ(std::get<0>(x), 2);
    EXPECT_EQ(std::get<1>(x), 2);
    // Check each vector element is equal to 2
    EXPECT_EQ(std::get<2>(x).size(), 3);
    for (int i = 0; i < std::get<2>(x).size(); ++i) {
      EXPECT_EQ(std::get<2>(x)(i), 2);
    }
    // Check each tuple element is equal to 2
    auto&& first_inner_tuple = std::get<3>(x);
    EXPECT_EQ(std::get<0>(first_inner_tuple), 2);
    auto&& first_inner_tuple_vector = std::get<1>(first_inner_tuple);
    EXPECT_EQ(first_inner_tuple_vector.size(), 1);
    auto&& first_inner_tuple_vector_tuple = first_inner_tuple_vector[0];
    EXPECT_EQ(std::get<0>(first_inner_tuple_vector_tuple), 2);
    for (int i = 0; i < std::get<2>(x).size(); ++i) {
      EXPECT_EQ(std::get<1>(first_inner_tuple_vector_tuple)(i), 2);
    }
  }
}

}