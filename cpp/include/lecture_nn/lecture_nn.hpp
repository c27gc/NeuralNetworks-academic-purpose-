#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace lecture_nn {

using Vector = std::vector<float>;
using Matrix = std::vector<Vector>;

class Dense {
public:
    explicit Dense(std::size_t units, std::string activation = "linear", bool use_bias = true);
    Dense(Dense&&) noexcept;
    Dense& operator=(Dense&&) noexcept;
    Dense(const Dense&) = delete;
    Dense& operator=(const Dense&) = delete;
    ~Dense();

    void build(std::size_t input_units);
    void set_weights(const Matrix& weights, const Vector& bias = {});
    Matrix forward(const Matrix& batch) const;
    std::size_t units() const noexcept;
    bool built() const noexcept;

private:
    std::size_t input_units_ = 0;
    std::size_t units_ = 0;
    std::string activation_;
    bool use_bias_ = true;
    Matrix weights_;
    Vector bias_;
    struct LegacyNeurons;
    std::unique_ptr<LegacyNeurons> legacy_neurons_;
};

class Sequential {
public:
    explicit Sequential(std::size_t input_units = 0);

    void add(Dense layer);
    void compile(const std::string& loss = "mse", const std::string& optimizer = "sgd");
    Matrix predict(const Matrix& batch) const;

private:
    std::size_t input_units_ = 0;
    bool compiled_ = false;
    std::vector<Dense> layers_;
};

} // namespace lecture_nn
