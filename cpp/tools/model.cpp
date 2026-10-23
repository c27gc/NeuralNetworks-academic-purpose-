#include "lecture_nn/lecture_nn.hpp"

#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
lecture_nn::Matrix csv(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open data file: " + path);
    lecture_nn::Matrix data;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        std::istringstream row(line);
        std::string cell;
        lecture_nn::Vector values;
        if (line.back() == ',') throw std::invalid_argument("empty CSV cell");
        while (std::getline(row, cell, ',')) {
            std::size_t consumed = 0;
            const float value = std::stof(cell, &consumed);
            if (cell.find_first_not_of(" \t\r", consumed) != std::string::npos)
                throw std::invalid_argument("invalid CSV number");
            values.push_back(value);
        }
        data.push_back(std::move(values));
    }
    return data;
}

std::size_t positive(const char* text) {
    const std::string value(text);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("epochs and batch size must be positive integers");
    const auto result = std::stoull(value);
    if (!result) throw std::invalid_argument("epochs and batch size must be positive integers");
    return result;
}

void worker(int argc, char** argv) {
    (void)argc;
    auto model = lecture_nn::Sequential::load(argv[2]);
    const auto inputs = csv(argv[3]), targets = csv(argv[4]), grid = csv(argv[5]);
    std::size_t epoch = 0;
    lecture_nn::Vector history{model.evaluate(inputs, targets).loss};
    std::string command;
    while (std::getline(std::cin, command)) {
        if (command == "quit") break;
        if (command != "snapshot") {
            std::istringstream request(command);
            std::string action, count, trailing;
            if (!(request >> action >> count) || action != "step" || (request >> trailing))
                throw std::invalid_argument("invalid worker command");
            const auto epochs = positive(count.c_str());
            if (epochs > 25 || epoch + epochs > 3000) throw std::invalid_argument("training limit reached");
            const auto result = model.fit(inputs, targets, epochs, 16);
            history.insert(history.end(), result.loss.begin(), result.loss.end());
            epoch += epochs;
        }
        const auto metrics = model.evaluate(inputs, targets);
        if (!std::isfinite(metrics.loss)) throw std::runtime_error("nonfinite training loss");
        const auto surface = model.predict(grid);
        std::cout << std::setprecision(9) << "{\"epoch\":" << epoch << ",\"history\":[";
        for (std::size_t i = 0; i < history.size(); ++i) std::cout << (i ? "," : "") << history[i];
        std::cout << "],\"metrics\":{\"loss\":" << metrics.loss << ",\"accuracy\":" << metrics.accuracy
                  << "},\"surface\":[";
        for (std::size_t i = 0; i < surface.size(); ++i) std::cout << (i ? "," : "") << surface[i][0];
        std::cout << "]}" << std::endl;
    }
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 6 && std::string(argv[1]) == "worker") {
            worker(argc, argv);
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "predict") {
            const auto model = lecture_nn::Sequential::load(argv[2]);
            const auto prediction = model.predict(csv(argv[3]));
            std::cout << std::setprecision(9);
            for (const auto& row : prediction) {
                for (std::size_t j = 0; j < row.size(); ++j) std::cout << (j ? "," : "") << row[j];
                std::cout << '\n';
            }
            return 0;
        }
        if (argc == 8 && std::string(argv[1]) == "train") {
            auto model = lecture_nn::Sequential::load(argv[2]);
            model.fit(csv(argv[3]), csv(argv[4]), positive(argv[6]), positive(argv[7]), false);
            model.save(argv[5]);
            return 0;
        }
        std::cerr << "Usage:\n  lecture_nn_model predict MODEL.nn INPUT.csv\n"
                  << "  lecture_nn_model train MODEL.nn INPUT.csv TARGET.csv OUTPUT.nn EPOCHS BATCH_SIZE\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
