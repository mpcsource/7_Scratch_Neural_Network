#include "gtest/gtest.h"
#include "math/matrix.h"
#include "math/scalar.h"
#include "network/neuron.h"

#define Scalar(t) std::make_shared<Scalar<t>>
#define Neuron(t) std::make_shared<Neuron<t>>

/*
TEST(A, MatrixOps)
{
    Matrix<int> m1(10, 1, 4);
    Matrix<int> m2(1, 10, 4);
    auto m3 = m1.dot(m2);
    m3.basicPrint();
    for(auto &g : m3.grad_) g = 1;
    m3.backward();
    for (auto &g : m2.grad_) std::cout << g << std::endl;
}

TEST(B, A)
{
    // A is 2×3, filled {1,2,3,4,5,6}
    // B is 3×2, filled {7,8,9,10,11,12}
    Matrix<int> A(2, 3, std::vector<int>{1, 2, 3, 4, 5, 6});
    Matrix<int> B(3, 2, std::vector<int>{7, 8, 9, 10, 11, 12});

    auto C = A.dot(B);
    C.basicPrint(); // should print a 2×2 matrix:
    // C[0,0] = 1*7 + 2*9 + 3*11 = 58
    // C[0,1] = 1*8 + 2*10+ 3*12 = 64
    // C[1,0] = 4*7 + 5*9 + 6*11 = 139
    // C[1,1] = 4*8 + 5*10+ 6*12 = 154

    C.backward(); // seed ∂L/∂C with ones

    // Now print grads of A (shape 2×3) and B (shape 3×2):
    std::cout << "A.grad:\n";
    for (auto g : A.grad_)
        std::cout << g << " ";
    std::cout << "\nB.grad:\n";
    for (auto g : B.grad_)
        std::cout << g << " ";
    std::cout << std::endl;
}
    */



TEST(A, B) {
    auto a = Scalar(double) (2.0);
    auto b = Scalar(double) (2);

    auto c = a->power(b);   // c = a * b
    c->grad = 1.0;       // seed gradient at the end
    c->_backward();

    std::cout << "c.data = " << c->data  << "\n"
              << "a.grad = " << a->grad  << "\n"
              << "b.grad = " << b->grad  << "\n";
}

TEST(A, C) {
    auto a = Neuron(double) (1);
}