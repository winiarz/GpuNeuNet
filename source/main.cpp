#include <iostream>
#include "Matrix.hpp"
#include "MatrixMultiplyTest.hpp"
#include "MultiMatrixMultiplyTest.hpp"

void testSingleMatrixMultiply()
{
    MatrixMultiplyTest test_simple(MatrixType_normal, MatrixType_normal, MatrixType_normal, "kernel/matrix_mult.cl", "matrixMultiply_simple_n_n_n");
    test_simple.performTest();
    MatrixMultiplyTest test_opt1(MatrixType_normal, MatrixType_normal, MatrixType_normal, "kernel/matrix_mult.cl", "matrixMultiply_opt1_n_n_n");
    test_opt1.performTest();
    MatrixMultiplyTest test_opt2(MatrixType_normal, MatrixType_normal, MatrixType_normal, "kernel/matrix_mult.cl", "matrixMultiply_opt2_n_n_n");
    test_opt2.performTest();

    MatrixMultiplyTest test_sw4(MatrixType_normal, MatrixType_swapped, MatrixType_normal, "kernel/matrix_mult.cl", "matrixMultiply_simple_n_s_n");
    test_sw4.performTest();
    MatrixMultiplyTest test_sw5(MatrixType_normal, MatrixType_swapped, MatrixType_normal, "kernel/matrix_mult.cl", "matrixMultiply_opt1_n_s_n");
    test_sw5.performTest();
}

int main()
{
    srand48(123456789);

    testSingleMatrixMultiply();


    MultiMatrix_SeparateNormal mm;
    MultiMatrixMultiplyTest multiTest(MultiMatrixType_separateNormal, MatrixType_normal, MultiMatrixType_separateNormal,
                                      "kernel/multi_matrix_mult.cl", "multiMatrixMultiply_simple_n_n_n");
    multiTest.performTest();
    MultiMatrixMultiplyTest multiTest2(MultiMatrixType_separateNormal, MatrixType_normal, MultiMatrixType_separateNormal,
                                       "kernel/multi_matrix_mult.cl", "multiMatrixMultiply_opt1_n_n_n");
    multiTest2.performTest();

}

