#include <iostream>
#include "Matrix.hpp"
#include "MatrixMultiplyTest.hpp"


int main()
{
    srand48(123456789);
    MatrixMultiplyTest test_simple(MatrixType_normal, MatrixType_normal, MatrixType_normal, "kernel/matrix_mult.cl", "matrixMultiply_simple_n_n_n");
    test_simple.performTest();
    MatrixMultiplyTest test_opt1(MatrixType_normal, MatrixType_normal, MatrixType_normal, "kernel/matrix_mult.cl", "matrixMultiply_opt1_n_n_n");
    test_opt1.performTest();
    MatrixMultiplyTest test_opt2(MatrixType_normal, MatrixType_normal, MatrixType_normal, "kernel/matrix_mult.cl", "matrixMultiply_opt2_n_n_n");
    test_opt2.performTest();

    /*MatrixMultiplyTest test_sw1(MatrixType_swapped, MatrixType_normal, MatrixType_normal, "kernel/matrix_mult.cl", "matrixMultiply_simple_s_n_n");
    test_sw1.performTest();
    MatrixMultiplyTest test_sw2(MatrixType_swapped, MatrixType_normal, MatrixType_normal, "kernel/matrix_mult.cl", "matrixMultiply_opt_s_n_n");
    test_sw2.performTest();
    MatrixMultiplyTest test_sw3(MatrixType_swapped, MatrixType_normal, MatrixType_normal, "kernel/matrix_mult.cl", "matrixMultiply_opt2_s_n_n");
    test_sw3.performTest();*/

    MatrixMultiplyTest test_sw4(MatrixType_normal, MatrixType_swapped, MatrixType_normal, "kernel/matrix_mult.cl", "matrixMultiply_simple_n_s_n");
    test_sw4.performTest();
    MatrixMultiplyTest test_sw5(MatrixType_normal, MatrixType_swapped, MatrixType_normal, "kernel/matrix_mult.cl", "matrixMultiply_opt1_n_s_n");
    test_sw5.performTest();
}

