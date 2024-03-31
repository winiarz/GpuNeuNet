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

//    testSingleMatrixMultiply();

    MultiMatrix_SeparateNormal mm;
    MultiMatrixMultiplyTest multiTest(MultiMatrixType_separateNormal, MatrixType_normal, MultiMatrixType_separateNormal,
                                      "kernel/multi_matrix_mult.cl", "multiMatrixMultiply_simple_n_n_n");
    multiTest.performTest();
    MultiMatrixMultiplyTest multiTest2(MultiMatrixType_separateNormal, MatrixType_normal, MultiMatrixType_separateNormal,
                                       "kernel/multi_matrix_mult.cl", "multiMatrixMultiply_opt1_n_n_n");
    multiTest2.performTest();

    MultiMatrixMultiplyTest multiTest3(MultiMatrixType_combinedNormal, MatrixType_normal, MultiMatrixType_combinedNormal,
                                       "kernel/multi_matrix_mult.cl", "multiMatrixMultiply_simple_cn_n_cn");
    multiTest3.performTest();

    MultiMatrixMultiplyTest multiTest4(MultiMatrixType_combinedNormal, MatrixType_normal, MultiMatrixType_combinedNormal,
                                       "kernel/multi_matrix_mult.cl", "multiMatrixMultiply_opt1_cn_n_cn");
    multiTest4.performTest();

    MultiMatrixMultiplyTest multiTest5(MultiMatrixType_combinedNormal, MatrixType_normal, MultiMatrixType_combinedNormal,
                                       "kernel/multi_matrix_mult.cl", "multiMatrixMultiply_opt2_cn_n_cn");
    multiTest5.performTest();
    MultiMatrixMultiplyTest multiTest6(MultiMatrixType_combinedNormal, MatrixType_normal, MultiMatrixType_combinedNormal,
                                       "kernel/multi_matrix_mult.cl", "multiMatrixMultiply_opt3_cn_n_cn");
    multiTest6.performTest();
    MultiMatrixMultiplyTest multiTest7(MultiMatrixType_combinedNormal, MatrixType_normal, MultiMatrixType_combinedNormal,
                                       "kernel/multi_matrix_mult.cl", "multiMatrixMultiply_opt4_cn_n_cn");
    multiTest7.performTest();
    MultiMatrixMultiplyTest multiTest8(MultiMatrixType_combinedNormal, MatrixType_normal, MultiMatrixType_combinedNormal,
                                       "kernel/multi_matrix_mult.cl", "multiMatrixMultiply_opt5_cn_n_cn");
    multiTest8.performTest();
    MultiMatrixMultiplyTest multiTest8a(MultiMatrixType_combinedNormal, MatrixType_normal, MultiMatrixType_combinedNormal,
                                       "kernel/multi_matrix_mult.cl", "multiMatrixMultiply_opt6_cn_n_cn");
    multiTest8a.performTest();

    MultiMatrixMultiplyTest multiTest9(MultiMatrixType_combinedNormal, MatrixType_swapped, MultiMatrixType_combinedNormal,
                                       "kernel/multi_matrix_mult_sw.cl", "multiMatrixMultiply_opt1_cn_s_cn");
    multiTest9.performTest();
    MultiMatrixMultiplyTest multiTest10(MultiMatrixType_combinedNormal, MatrixType_swapped, MultiMatrixType_combinedNormal,
                                        "kernel/multi_matrix_mult_sw.cl", "multiMatrixMultiply_opt2_cn_s_cn");
    multiTest10.performTest();
    MultiMatrixMultiplyTest multiTest11(MultiMatrixType_combinedNormal, MatrixType_swapped, MultiMatrixType_combinedNormal,
                                        "kernel/multi_matrix_mult_sw.cl", "multiMatrixMultiply_opt3_cn_s_cn");
    multiTest11.performTest();

}

