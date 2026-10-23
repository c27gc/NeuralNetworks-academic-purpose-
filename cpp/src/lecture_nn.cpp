#include "lecture_nn/lecture_nn.hpp"
#include "neuron.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace lecture_nn {
namespace {

float activate(const std::string& name, float value) {
    if (name == "linear") return value;
    if (name == "sigmoid") {
        const float z = std::exp(-std::fabs(value));
        return value >= 0 ? 1 / (1 + z) : z / (1 + z);
    }
    if (name == "relu") return std::max(0.0F, value);
    if (name == "tanh") return std::tanh(value);
    throw std::invalid_argument("unsupported activation: " + name);
}

float derivative(const std::string& name, float output) {
    if (name == "sigmoid") return output * (1 - output);
    if (name == "tanh") return 1 - output * output;
    if (name == "relu") return output > 0 ? 1.0F : 0.0F;
    return 1;
}

void validate_matrix(const Matrix& matrix, std::size_t width, bool empty = false) {
    if (!empty && matrix.empty()) throw std::invalid_argument("matrix must not be empty");
    for (const auto& row : matrix) {
        if (row.size() != width) throw std::invalid_argument("inconsistent matrix width");
        for (float value : row)
            if (!std::isfinite(value)) throw std::invalid_argument("matrix must contain finite values");
    }
}

template <typename T> T read(std::istream& input) {
    T value;
    if (!(input >> value)) throw std::invalid_argument("invalid or truncated model file");
    return value;
}

} // namespace

struct Dense::NeuronBank {
    std::vector<std::unique_ptr<NeuronClass>> values;
    std::mutex mutex;
};

Dense::Dense(std::size_t units, std::string activation, bool use_bias)
    : units_(units), activation_(std::move(activation)), use_bias_(use_bias) {
    if (!units_) throw std::invalid_argument("units must be positive");
    (void)activate(activation_, 0);
}
Dense::Dense(Dense&&) noexcept = default;
Dense& Dense::operator=(Dense&&) noexcept = default;
Dense::~Dense() = default;

void Dense::build(std::size_t input_units, unsigned seed) {
    if (!input_units) throw std::invalid_argument("input_units must be positive");
    std::mt19937 rng(seed);
    const float limit = std::sqrt(6.0F / static_cast<float>(input_units + units_));
    std::uniform_real_distribution<float> distribution(-limit, limit);
    Matrix weights(input_units, Vector(units_));
    for (auto& row : weights) for (auto& value : row) value = distribution(rng);
    set_weights(weights);
}

void Dense::set_weights(const Matrix& weights, const Vector& bias) {
    validate_matrix(weights, units_);
    if (weights.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("too many input features");
    Vector values = bias.empty() ? Vector(units_, 0) : bias;
    validate_matrix({values}, units_);
    auto neurons = std::make_unique<NeuronBank>();
    if (activation_ == "sigmoid") {
        for (std::size_t unit = 0; unit < units_; ++unit) {
            Vector column;
            for (const auto& row : weights) column.push_back(row[unit]);
            neurons->values.push_back(std::make_unique<NeuronClass>(
                static_cast<int>(weights.size()), column.data(), use_bias_ ? values[unit] : 0));
        }
    }
    input_units_ = weights.size();
    weights_ = weights;
    bias_ = use_bias_ ? values : Vector(units_, 0);
    neurons_ = std::move(neurons);
}

Matrix Dense::affine(const Matrix& batch) const {
    Matrix result(batch.size(), Vector(units_, 0));
    for (std::size_t n = 0; n < batch.size(); ++n) {
        for (std::size_t j = 0; j < units_; ++j) {
            result[n][j] = use_bias_ ? bias_[j] : 0;
            for (std::size_t i = 0; i < input_units_; ++i) result[n][j] += batch[n][i] * weights_[i][j];
        }
    }
    return result;
}

Matrix Dense::forward(const Matrix& batch) const {
    if (!built()) throw std::logic_error("build the layer before evaluation");
    validate_matrix(batch, input_units_, true);
    if (activation_ == "sigmoid") {
        std::lock_guard<std::mutex> lock(neurons_->mutex);
        Matrix output;
        for (auto sample : batch) {
            Vector row;
            for (auto& neuron : neurons_->values) {
                neuron->setInputsValues(sample.data());
                neuron->informationSpread();
                row.push_back(neuron->getOutputValue());
            }
            output.push_back(std::move(row));
        }
        return output;
    }
    Matrix result = affine(batch);
    for (auto& row : result) for (auto& value : row) value = activate(activation_, value);
    return result;
}

std::size_t Dense::units() const noexcept { return units_; }
bool Dense::built() const noexcept { return input_units_ != 0; }

Sequential::Sequential(std::size_t input_units, unsigned seed) : input_units_(input_units), rng_(seed) {}

void Sequential::add(Dense layer) {
    layers_.push_back(std::move(layer));
    compiled_ = false;
}

void Sequential::compile(const std::string& loss, const std::string& optimizer, float learning_rate) {
    compiled_ = false;
    if (layers_.empty()) throw std::invalid_argument("a model needs at least one layer");
    if (loss != "mse" && loss != "binary_crossentropy") throw std::invalid_argument("unsupported loss");
    if (optimizer != "sgd" && optimizer != "adam") throw std::invalid_argument("unsupported optimizer");
    if (!(learning_rate > 0) || !std::isfinite(learning_rate)) throw std::invalid_argument("invalid learning rate");
    std::size_t width = input_units_ ? input_units_ : layers_.front().input_units_;
    if (!width) throw std::invalid_argument("input shape is required");
    input_units_ = width;
    first_moment_.clear();
    second_moment_.clear();
    for (auto& layer : layers_) {
        if (!layer.built()) layer.build(width, rng_());
        if (layer.input_units_ != width) throw std::invalid_argument("layer input width mismatch");
        const auto count = width * layer.units_ + layer.units_;
        first_moment_.emplace_back(count, 0);
        second_moment_.emplace_back(count, 0);
        width = layer.units_;
    }
    if (loss == "binary_crossentropy" && layers_.back().activation_ != "sigmoid")
        throw std::invalid_argument("binary_crossentropy requires a sigmoid output");
    loss_ = loss;
    optimizer_ = optimizer;
    learning_rate_ = learning_rate;
    step_ = 0;
    compiled_ = true;
}

void Sequential::require_compiled() const {
    if (!compiled_) throw std::logic_error("compile the model first");
}

void Sequential::validate_data(const Matrix& x, const Matrix& y) const {
    require_compiled();
    validate_matrix(x, input_units_);
    validate_matrix(y, layers_.back().units_);
    if (x.size() != y.size()) throw std::invalid_argument("sample counts do not match");
    if (loss_ == "binary_crossentropy") for (const auto& row : y) for (float value : row)
        if (value < 0 || value > 1) throw std::invalid_argument("binary targets must be in [0, 1]");
}

Matrix Sequential::predict(const Matrix& batch) const {
    require_compiled();
    Matrix output = batch;
    for (const auto& layer : layers_) output = layer.forward(output);
    return output;
}

Metrics Sequential::evaluate(const Matrix& x, const Matrix& y) const {
    validate_data(x, y);
    Matrix previous = x;
    for (std::size_t i = 0; i + 1 < layers_.size(); ++i) previous = layers_[i].forward(previous);
    const auto prediction = layers_.back().forward(previous);
    const auto logits = layers_.back().affine(previous);
    double total = 0, correct = 0;
    const auto count = static_cast<double>(y.size() * layers_.back().units_);
    for (std::size_t n = 0; n < y.size(); ++n) for (std::size_t j = 0; j < y[n].size(); ++j) {
        if (loss_ == "mse") total += std::pow(prediction[n][j] - y[n][j], 2);
        else {
            const double z = logits[n][j];
            total += std::max(z, 0.0) - z * y[n][j] + std::log1p(std::exp(-std::fabs(z)));
            correct += (prediction[n][j] >= 0.5F) == (y[n][j] >= 0.5F);
        }
    }
    return {static_cast<float>(total / count), static_cast<float>(correct / count)};
}

std::vector<Gradient> Sequential::gradients(const Matrix& x, const Matrix& y) const {
    validate_data(x, y);
    std::vector<Matrix> activations{x};
    for (const auto& layer : layers_) activations.push_back(layer.forward(activations.back()));
    Matrix delta = activations.back();
    const float count = static_cast<float>(y.size() * layers_.back().units_);
    for (std::size_t n = 0; n < y.size(); ++n) for (std::size_t j = 0; j < y[n].size(); ++j) {
        const float output = delta[n][j];
        delta[n][j] = (output - y[n][j]) / count * (loss_ == "binary_crossentropy" ? 1 :
            2 * derivative(layers_.back().activation_, output));
    }
    std::vector<Gradient> result(layers_.size());
    for (std::size_t index = layers_.size(); index-- > 0;) {
        const auto& layer = layers_[index];
        auto& gradient = result[index];
        gradient.weights = Matrix(layer.input_units_, Vector(layer.units_, 0));
        gradient.bias = Vector(layer.units_, 0);
        for (std::size_t n = 0; n < x.size(); ++n) for (std::size_t j = 0; j < layer.units_; ++j) {
            if (layer.use_bias_) gradient.bias[j] += delta[n][j];
            for (std::size_t i = 0; i < layer.input_units_; ++i)
                gradient.weights[i][j] += activations[index][n][i] * delta[n][j];
        }
        if (index) {
            Matrix next(x.size(), Vector(layer.input_units_, 0));
            for (std::size_t n = 0; n < x.size(); ++n) for (std::size_t i = 0; i < layer.input_units_; ++i) {
                for (std::size_t j = 0; j < layer.units_; ++j) next[n][i] += delta[n][j] * layer.weights_[i][j];
                next[n][i] *= derivative(layers_[index - 1].activation_, activations[index][n][i]);
            }
            delta = std::move(next);
        }
    }
    return result;
}

void Sequential::update(const std::vector<Gradient>& gradients) {
    ++step_;
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        auto& layer = layers_[index];
        auto weights = layer.weights_;
        auto bias = layer.bias_;
        std::size_t offset = 0;
        auto update_value = [&](float& value, float gradient) {
            if (optimizer_ == "adam") {
                float& m = first_moment_[index][offset];
                float& v = second_moment_[index][offset];
                m = 0.9F * m + 0.1F * gradient;
                v = 0.999F * v + 0.001F * gradient * gradient;
                gradient = static_cast<float>((m / (1 - std::pow(0.9, step_))) /
                    (std::sqrt(v / (1 - std::pow(0.999, step_))) + 1e-8));
            }
            value -= learning_rate_ * gradient;
            if (!std::isfinite(value)) throw std::runtime_error("nonfinite parameters; reduce learning rate");
            ++offset;
        };
        for (std::size_t i = 0; i < weights.size(); ++i) for (std::size_t j = 0; j < layer.units_; ++j)
            update_value(weights[i][j], gradients[index].weights[i][j]);
        for (std::size_t j = 0; j < layer.units_; ++j) update_value(bias[j], gradients[index].bias[j]);
        layer.set_weights(weights, bias);
    }
}

History Sequential::fit(const Matrix& x, const Matrix& y, std::size_t epochs, std::size_t batch_size, bool shuffle) {
    validate_data(x, y);
    if (!epochs || !batch_size) throw std::invalid_argument("epochs and batch_size must be positive");
    std::vector<std::size_t> indices(x.size());
    std::iota(indices.begin(), indices.end(), 0);
    History history;
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        if (shuffle) std::shuffle(indices.begin(), indices.end(), rng_);
        for (std::size_t start = 0; start < x.size(); start += batch_size) {
            Matrix batch_x, batch_y;
            for (std::size_t i = start; i < std::min(start + batch_size, x.size()); ++i) {
                batch_x.push_back(x[indices[i]]);
                batch_y.push_back(y[indices[i]]);
            }
            update(gradients(batch_x, batch_y));
        }
        history.loss.push_back(evaluate(x, y).loss);
    }
    return history;
}

void Sequential::save(const std::string& path) const {
    require_compiled();
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot open model file for writing");
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    output << "LECTURE_NN 1\n" << input_units_ << ' ' << layers_.size() << '\n';
    output << loss_ << ' ' << optimizer_ << ' ' << learning_rate_ << '\n';
    for (const auto& layer : layers_) {
        output << layer.units_ << ' ' << layer.activation_ << ' ' << layer.use_bias_ << '\n';
        for (const auto& row : layer.weights_) for (float value : row) output << value << ' ';
        output << '\n';
        for (float value : layer.bias_) output << value << ' ';
        output << '\n';
    }
    output.flush();
    if (!output) throw std::runtime_error("could not write model file");
}

Sequential Sequential::load(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open model file");
    const auto magic = read<std::string>(input);
    const auto version = read<int>(input);
    if (magic != "LECTURE_NN" || version != 1) throw std::invalid_argument("unsupported model format");
    auto width = read<std::size_t>(input);
    const auto count = read<std::size_t>(input);
    if (!width || width > 4096 || !count || count > 128) throw std::invalid_argument("invalid model dimensions");
    const auto loss = read<std::string>(input);
    const auto optimizer = read<std::string>(input);
    const auto rate = read<float>(input);
    Sequential model(width);
    for (std::size_t i = 0; i < count; ++i) {
        const auto units = read<std::size_t>(input);
        const auto activation = read<std::string>(input);
        const auto bias_flag = read<int>(input);
        if (!units || units > 4096 || width * units > 1000000 || (bias_flag != 0 && bias_flag != 1))
            throw std::invalid_argument("invalid layer dimensions or bias flag");
        Matrix weights(width, Vector(units));
        for (auto& row : weights) for (auto& value : row) value = read<float>(input);
        Vector bias(units);
        for (auto& value : bias) value = read<float>(input);
        Dense layer(units, activation, bias_flag == 1);
        layer.set_weights(weights, bias);
        model.add(std::move(layer));
        width = units;
    }
    std::string trailing;
    if (input >> trailing) throw std::invalid_argument("unexpected data after model");
    model.compile(loss, optimizer, rate);
    return model;
}

std::string Sequential::summary() const {
    require_compiled();
    std::ostringstream output;
    std::size_t total = 0;
    for (const auto& layer : layers_) {
        const auto parameters = layer.input_units_ * layer.units_ + (layer.use_bias_ ? layer.units_ : 0);
        total += parameters;
        output << "Dense " << layer.input_units_ << " -> " << layer.units_ << "  " << layer.activation_
               << "  parameters=" << parameters << '\n';
    }
    output << "Total parameters: " << total;
    return output.str();
}

} // namespace lecture_nn
