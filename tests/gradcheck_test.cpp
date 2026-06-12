#include <iostream>
#include "CorePP.h"
#include "Autograd.h"
using namespace std;
using namespace CoreNNSpace;

int main() {
    cout << "\n===== Gradient Check: Linear Layer =====" << endl;

    Linear linear(4, 3, XavierUniform);
    Matrix<float> input(4, 1);
    InitMatrixFunc(input, Uniform, {.a = -1, .b = 1});
    Matrix<float> output = linear.forward(input);

    auto loss_fn = [&linear](const Matrix<float>& x) -> float {
        Matrix<float> xc = x;
        Matrix<float> out = linear.forward(xc);
        float loss = 0;
        for (int i = 0; i < out.row; ++i)
            for (int j = 0; j < out.col; ++j)
                loss += out.at(i, j) * out.at(i, j);
        return loss;
    };

    Matrix<float> num_grad = numerical_gradient<float>(loss_fn, input, 1e-4f);
    Matrix<float> grad_output(output.row, output.col);
    for (int i = 0; i < output.row; ++i)
        for (int j = 0; j < output.col; ++j)
            grad_output.at(i, j) = 2.0f * output.at(i, j);
    Matrix<float> analytical_grad = linear.backward(grad_output);

    num_grad.Analysis("Numerical dL/dx");
    analytical_grad.Analysis("Analytical dL/dx");

    bool ok = gradcheck(analytical_grad, num_grad, 1e-2f, "Linear dL/dx");
    cout << (ok ? "Gradient check PASSED!" : "Gradient check FAILED!") << endl;
    return ok ? 0 : 1;
}
