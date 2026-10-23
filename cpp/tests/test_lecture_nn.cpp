#include "lecture_nn/lecture_nn.hpp"

#include <cassert>
#include <cmath>
#include <utility>

int main() {
    lecture_nn::Dense sigmoid(1, "sigmoid");
    sigmoid.set_weights({{1.0F}}, {0.0F});
    auto one = sigmoid.forward({{0.0F}});
    assert(std::fabs(one[0][0] - 0.5F) < 1e-5F);

    lecture_nn::Sequential model(2);
    lecture_nn::Dense first(2);
    first.set_weights({{1.0F, 0.0F}, {0.0F, 1.0F}});
    lecture_nn::Dense second(1, "sigmoid");
    second.set_weights({{1.0F}, {1.0F}}, {0.0F});
    model.add(std::move(first));
    model.add(std::move(second));
    model.compile();

    auto prediction = model.predict({{1.0F, 1.0F}});
    const float expected = 1.0F / (1.0F + std::exp(-2.0F));
    assert(std::fabs(prediction[0][0] - expected) < 1e-5F);
    return 0;
}
