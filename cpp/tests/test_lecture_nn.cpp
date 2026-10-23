#include "lecture_nn/lecture_nn.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

using lecture_nn::Dense;
using lecture_nn::Matrix;
using lecture_nn::Sequential;
using lecture_nn::Vector;

namespace {
int checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
template <typename Function> void rejects(Function function) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "expected invalid input to be rejected");
}
void near(float a, float b, float tolerance = 1e-5F) {
    require(std::isfinite(a) && std::isfinite(b) && std::fabs(a - b) < tolerance, "numeric comparison failed");
}

Sequential configured(const Matrix& weights, const Vector& bias, const std::string& activation,
                      const std::string& loss = "mse", const std::string& optimizer = "sgd") {
    Sequential model(2);
    Dense hidden(2, activation);
    hidden.set_weights(weights, bias);
    model.add(std::move(hidden));
    Dense output(1, loss == "mse" ? "linear" : "sigmoid");
    output.set_weights({{0.4F}, {-0.3F}}, {0.2F});
    model.add(std::move(output));
    model.compile(loss, optimizer, 0.03F);
    return model;
}

void gradients() {
    const Matrix x{{0.3F, -0.2F}, {0.8F, 0.4F}}, y{{0.1F}, {0.7F}};
    for (const auto& activation : {"linear", "tanh", "sigmoid", "relu"}) {
        for (const auto& loss : {"mse", "binary_crossentropy"}) {
            Matrix weights{{0.4F, -0.3F}, {0.1F, 0.6F}};
            Vector bias{0.2F, 0.3F};
            auto model = configured(weights, bias, activation, loss);
            const auto analytical = model.gradients(x, y);
            const float epsilon = 0.001F;
            for (std::size_t i = 0; i < 2; ++i) for (std::size_t j = 0; j < 2; ++j) {
                const float saved = weights[i][j];
                weights[i][j] = saved + epsilon;
                const float plus = configured(weights, bias, activation, loss).evaluate(x, y).loss;
                weights[i][j] = saved - epsilon;
                const float minus = configured(weights, bias, activation, loss).evaluate(x, y).loss;
                weights[i][j] = saved;
                near(analytical[0].weights[i][j], (plus - minus) / (2 * epsilon), 0.0002F);
            }
            for (std::size_t j = 0; j < 2; ++j) {
                const float saved = bias[j];
                bias[j] = saved + epsilon;
                const float plus = configured(weights, bias, activation, loss).evaluate(x, y).loss;
                bias[j] = saved - epsilon;
                const float minus = configured(weights, bias, activation, loss).evaluate(x, y).loss;
                bias[j] = saved;
                near(analytical[0].bias[j], (plus - minus) / (2 * epsilon), 0.0002F);
            }
        }
    }
}

void regression() {
    const Matrix x{{-1}, {0}, {1}, {2}, {3}}, y{{-1}, {1}, {3}, {5}, {7}};
    Sequential model(1);
    model.add(Dense(1));
    model.compile("mse", "sgd", 0.03F);
    model.fit(x, y, 300, 2);
    require(model.evaluate(x, y).loss < 1e-6F, "SGD regression did not converge");
    near(model.predict({{4}})[0][0], 9, 0.001F);
}

void contracts() {
    auto model = configured({{0.4F, -0.3F}, {0.1F, 0.6F}}, {0.2F, 0.3F}, "tanh");
    rejects([&] { model.fit({}, {}); });
    rejects([&] { model.fit({{1, 2}}, {{0}, {1}}); });
    rejects([&] { model.predict({{1}}); });
    rejects([&] { model.predict({{std::numeric_limits<float>::quiet_NaN(), 2}}); });
    rejects([&] { model.fit({{1, 2}}, {{0}}, 0); });
    rejects([&] { model.fit({{1, 2}}, {{0}}, 2, 0); });
    const auto before = model.predict({{0, 0}});
    rejects([&] { model.compile("mse", "sgd", -1); });
    rejects([&] { model.predict({{0, 0}}); });
    model.compile();
    near(model.predict({{0, 0}})[0][0], before[0][0]);
    model.add(Dense(1));
    rejects([&] { model.predict({{0, 0}}); });
    model.compile();
    Dense mismatch(1);
    mismatch.set_weights({{1}, {2}});
    model.add(std::move(mismatch));
    rejects([&] { model.compile(); });
    Dense no_bias(1, "linear", false);
    no_bias.set_weights({{2}}, {99});
    near(no_bias.forward({{3}})[0][0], 6);
    rejects([&] { no_bias.set_weights({{2}}, {1, 2}); });
    near(no_bias.forward({{3}})[0][0], 6);
    Sequential saturated(1);
    Dense sigmoid(1, "sigmoid");
    sigmoid.set_weights({{1000}});
    saturated.add(std::move(sigmoid));
    saturated.compile("binary_crossentropy");
    near(saturated.evaluate({{1}}, {{0}}).loss, 1000);
    near(saturated.gradients({{1}}, {{0}})[0].weights[0][0], 1);
    rejects([&] { saturated.fit({{1}}, {{2}}); });
    require(saturated.predict({}).empty(), "empty inference should return an empty matrix");
}

void persistence() {
    auto model = configured({{0.4F, -0.3F}, {0.1F, 0.6F}}, {0.2F, 0.3F}, "tanh");
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() / ("lecture-nn-test-" + std::to_string(stamp) + ".nn");
    model.save(path.string());
    auto restored = Sequential::load(path.string());
    near(model.predict({{0.3F, 0.7F}})[0][0], restored.predict({{0.3F, 0.7F}})[0][0]);
    {
        std::ofstream file(path);
        file << "LECTURE_NN 1 2 1 mse sgd 0.1 2 tanh 1 0";
    }
    rejects([&] { Sequential::load(path.string()); });
    std::filesystem::remove(path);
}
}

int main() {
    try {
        gradients();
        regression();
        contracts();
        persistence();
        std::cout << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Failed after " << checks << " checks: " << error.what() << '\n';
        return 1;
    }
}
