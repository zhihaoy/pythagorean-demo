#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xtensor.hpp>

static xt::xarray<int>
next_pytharogian_triples(xt::xarray<int> const &previous_stage) {
  xt::xarray<int> stacked_matrices = {{-1, 2, 2}, {-2, 1, 2}, {-2, 2, 3},
                                      {1, 2, 2},  {2, 1, 2},  {2, 2, 3},
                                      {1, -2, 2}, {2, -1, 2}, {2, -2, 3}};

  auto shape = previous_stage.shape();
  xt::xarray<int> next_three = xt::transpose(
      xt::linalg::dot(stacked_matrices, xt::transpose(previous_stage)));
  next_three.reshape({3 * shape[0], shape[1]});
  return next_three;
}

#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  int n = argc < 2 ? 10 : std::stoi(argv[1]);

  xt::xarray<int> current = {{3, 4, 5}};
  for (int i = 0;;) {
    for (int row = 0; row < current.shape()[0]; ++row) {
      if (i++ < n)
        std::cout << xt::view(current, row) << '\n';
      else
        return 0;
    }
    current = next_pytharogian_triples(current);
  }
}
