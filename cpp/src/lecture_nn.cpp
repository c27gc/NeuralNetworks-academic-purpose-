#include "lecture_nn/lecture_nn.hpp"

#include "neuron.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace lecture_nn {

namespace {

float activate(const std::string& name, float value) {
    if (name == "linear") return value;
    if (name == "sigmoid") return 1.0F / (1.0F + std::exp(-value));
    if (name == "relu") return value > 0.0F ? value : 0.0F;
    throw std::invalid_argument("unsupported activation: " + name);
}

void validate_activation(const std::string& name) {
    (void)activate(name, 0.0F);
}

Vector column(const Matrix& matrix, std::size_t index) {
    Vector values;
    values.reserve(matrix.size());
    for (const auto& row : matrix) values.push_back(row[index]);
    return values;
}

} // namespace

struct Dense::LegacyNeurons {
    std::vector<std::unique_ptr<NeuronClass>> values;
};

Dense::Dense(std::size_t units, std::string activation, bool use_bias)
    : units_(units), activation_(std::move(activation)), use_bias_(use_bias) {
    if (units_ == 0) throw std::invalid_argument("units must be positive");
    validate_activation(activation_);
}

Dense::Dense(Dense&&) noexcept = default;
Dense& Dense::operator=(Dense&&) noexcept = default;
Dense::~Dense() = default;

void Dense::build(std::size_t input_units) {
    if (input_units == 0) throw std::invalid_argument("input_units must be positive");
    input_units_ = input_units;
    weights_ = Matrix(input_units_, Vector(units_, 0.0F));
    bias_ = Vector(units_, 0.0F);
    legacy_neurons_ = std::make_unique<LegacyNeurons>();
    legacy_neurons_->values.reserve(units_);
    for (std::size_t unit = 0; unit < units_; ++unit) {
        legacy_neurons_->values.push_back(
            std::make_unique<NeuronClass>(static_cast<int>(input_units_), column(weights_, unit).data(), use_bias_ ? bias_[unit] : 0.0F));
    }
}

void Dense::set_weights(const Matrix& weights, const Vector& bias) {
    if (weights.empty() || weights.front().size() != units_) {
        throw std::invalid_argument("weights must have shape (input_units, units)");
    }
    for (const auto& row : weights) {
        if (row.size() != units_) throw std::invalid_argument("inconsistent weight matrix");
    }
    input_units_ = weights.size();
    weights_ = weights;
    bias_ = bias.empty() ? Vector(units_, 0.0F) : bias;
    if (bias_.size() != units_) throw std::invalid_argument("bias must have one value per unit");
    legacy_neurons_ = std::make_unique<LegacyNeurons>();
    legacy_neurons_->values.reserve(units_);
    for (std::size_t unit = 0; unit < units_; ++unit) {
        auto weights_column = column(weights_, unit);
        legacy_neurons_->values.push_back(
            std::make_unique<NeuronClass>(static_cast<int>(input_units_), weights_column.data(), use_bias_ ? bias_[unit] : 0.0F));
    }
}

Matrix Dense::forward(const Matrix& batch) const {
    if (input_units_ == 0 || !legacy_neurons_) throw std::logic_error("layer must be built before evaluation");
    Matrix result;
    result.reserve(batch.size());
    for (const auto& sample : batch) {
        if (sample.size() != input_units_) throw std::invalid_argument("sample width does not match layer input");
        Vector output(units_, 0.0F);
        Vector sample_copy = sample;
        for (std::size_t unit = 0; unit < units_; ++unit) {
            float total = use_bias_ ? bias_[unit] : 0.0F;
            for (std::size_t input = 0; input < input_units_; ++input) total += sample[input] * weights_[input][unit];
            if (activation_ == "sigmoid") {
                auto& neuron = *legacy_neurons_->values[unit];
                neuron.setInputsValues(sample_copy.data());
                neuron.informationSpread();
                output[unit] = neuron.getOutputValue();
            } else {
                output[unit] = activate(activation_, total);
            }
        }
        result.push_back(std::move(output));
    }
    return result;
}

std::size_t Dense::units() const noexcept { return units_; }
bool Dense::built() const noexcept { return input_units_ != 0 && legacy_neurons_ != nullptr; }

Sequential::Sequential(std::size_t input_units) : input_units_(input_units) {}

void Sequential::add(Dense layer) {
    if (!layer.built()) {
        if (!layers_.empty()) layer.build(layers_.back().units());
        else if (input_units_ != 0) layer.build(input_units_);
    }
    layers_.push_back(std::move(layer));
}

void Sequential::compile(const std::string& loss, const std::string& optimizer) {
    if (layers_.empty()) throw std::logic_error("a model needs at least one layer");
    if (loss != "mse") throw std::invalid_argument("only mse is available in the reference implementation");
    if (optimizer != "sgd" && optimizer != "adam") throw std::invalid_argument("unsupported optimizer");
    compiled_ = true;
}

Matrix Sequential::predict(const Matrix& batch) const {
    if (!compiled_) throw std::logic_error("compile the model before calling predict");
    Matrix result = batch;
    for (const auto& layer : layers_) result = layer.forward(result);
    return result;
}

} // namespace lecture_nn
