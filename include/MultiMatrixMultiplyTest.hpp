#pragma once
#include "MultiMatrix.hpp"
#include <string>
#include "ClKernelFromSourceLoader.hpp"

class MultiMatrixMultiplyTest {
public:
    MultiMatrixMultiplyTest(MultiMatrixType, MatrixType, MultiMatrixType, std::string, std::string);

    bool performTest();
private:
    void prepareTest();
    std::shared_ptr<IMatrix> createMatrixOfType(MatrixType);
    std::shared_ptr<IMultiMatrix> createMultiMatrixOfType(MultiMatrixType);

    std::shared_ptr<IMatrix> B;
    std::shared_ptr<IMultiMatrix> A,C,C_k;

    std::string fileName;
    std::string kernelName;
    static std::shared_ptr<ClKernelFromSourceLoader> kernelLoader;
    std::vector<std::shared_ptr<ClMemory>> gpuMemory;
    std::shared_ptr<ClKernel> kernel;
};

