#pragma once

#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace lecture_nn {

using Vector = std::vector<float>;
using Matrix = std::vector<Vector>;

struct Gradient { Matrix weights; Vector bias; };
struct Metrics { float loss = 0; float accuracy = 0; };
struct History { Vector loss; };

class Dense {
public:
    explicit Dense(std::size_t units, std::string activation = "linear", bool use_bias = true);
    Dense(Dense&&) noexcept;
    Dense& operator=(Dense&&) noexcept;
    Dense(const Dense&) = delete;
    Dense& operator=(const Dense&) = delete;
    ~Dense();

    void build(std::size_t input_units, unsigned seed = 42);
    void set_weights(const Matrix& weights, const Vector& bias = {});
    Matrix forward(const Matrix& batch) const;
    std::size_t units() const noexcept;
    bool built() const noexcept;
    const Matrix& weights() const noexcept { return weights_; }
    const Vector& bias() const noexcept { return bias_; }

private:
    friend class Sequential;
    Matrix affine(const Matrix& batch) const;
    std::size_t input_units_ = 0;
    std::size_t units_;
    std::string activation_;
    bool use_bias_;
    Matrix weights_;
    Vector bias_;
    struct NeuronBank;
    std::unique_ptr<NeuronBank> neurons_;
};

class Sequential {
public:
    explicit Sequential(std::size_t input_units = 0, unsigned seed = 42);
    void add(Dense layer);
    void compile(const std::string& loss = "mse", const std::string& optimizer = "sgd",
                 float learning_rate = 0.01F);
    Matrix predict(const Matrix& batch) const;
    Metrics evaluate(const Matrix& inputs, const Matrix& targets) const;
    std::vector<Gradient> gradients(const Matrix& inputs, const Matrix& targets) const;
    History fit(const Matrix& inputs, const Matrix& targets, std::size_t epochs = 100,
                std::size_t batch_size = 32, bool shuffle = true);
    void save(const std::string& path) const;
    static Sequential load(const std::string& path);
    std::string summary() const;

private:
    void require_compiled() const;
    void validate_data(const Matrix& x, const Matrix& y) const;
    void update(const std::vector<Gradient>& gradients);
    std::size_t input_units_;
    bool compiled_ = false;
    std::string loss_, optimizer_;
    float learning_rate_ = 0.01F;
    std::size_t step_ = 0;
    std::mt19937 rng_;
    std::vector<Dense> layers_;
    std::vector<Vector> first_moment_, second_moment_;
};

} // namespace lecture_nn
