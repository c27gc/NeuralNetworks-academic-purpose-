#include "lecture_nn/lecture_nn.hpp"
#include <iostream>

int main(int argc, char** argv) {
    const lecture_nn::Matrix x{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const lecture_nn::Matrix y{{0}, {1}, {1}, {0}};
    lecture_nn::Sequential model(2, 42);
    model.add(lecture_nn::Dense(8, "tanh"));
    model.add(lecture_nn::Dense(1, "sigmoid"));
    model.compile("binary_crossentropy", "adam", 0.03F);
    std::cout << model.summary() << "\nInitial loss: " << model.evaluate(x, y).loss << '\n';
    model.fit(x, y, 800, 4, false);
    const auto metrics = model.evaluate(x, y);
    std::cout << "Loss: " << metrics.loss << "  Accuracy: " << metrics.accuracy << '\n';
    const auto prediction = model.predict(x);
    for (std::size_t i = 0; i < x.size(); ++i)
        std::cout << x[i][0] << " xor " << x[i][1] << " = " << prediction[i][0] << '\n';
    if (argc == 2) model.save(argv[1]);
    return metrics.accuracy == 1 && metrics.loss < 0.05F ? 0 : 1;
}
