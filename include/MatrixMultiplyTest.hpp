#pragma once
#include <string>
#include "Matrix.hpp"
#include "ClKernelFromSourceLoader.hpp"

class MatrixMultiplyTest {
public:
    MatrixMultiplyTest(MatrixType, MatrixType, MatrixType, std::string, std::string);

    bool performTest();
private:
    void prepareTest();
    std::shared_ptr<IMatrix> createMatrixOfType(MatrixType);

    std::string kernelSourceFilename, kernelName;
    std::shared_ptr<IMatrix> A,B,C_k;
    Matrix C;
    boost::shared_ptr<ClKernel> kernel;
    static std::shared_ptr<ClKernelFromSourceLoader> kernelLoader;
    std::vector<std::shared_ptr<ClMemory>> gpuMemory;
};


