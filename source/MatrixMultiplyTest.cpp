#include "MatrixMultiplyTest.hpp"
#include "time.hpp"
#include "ClKernel.hpp"

std::shared_ptr<ClKernelFromSourceLoader> MatrixMultiplyTest::kernelLoader = nullptr;

MatrixMultiplyTest::MatrixMultiplyTest(MatrixType typeLeft, MatrixType typeRight, MatrixType typeResult,
                                       std::string p_kernelSourceFilename, std::string p_kernelName) :
    kernelSourceFilename(p_kernelSourceFilename),
    kernelName(p_kernelName)
{
    A = createMatrixOfType(typeLeft);
    B = createMatrixOfType(typeRight);
    C_k = createMatrixOfType(typeResult);

    if(kernelLoader == nullptr)
    {
        std::set<std::string> clIncludeDirs;
        kernelLoader = std::make_shared<ClKernelFromSourceLoader>(clIncludeDirs);
    }
}

std::shared_ptr<IMatrix> MatrixMultiplyTest::createMatrixOfType(MatrixType matrixType)
{
    switch(matrixType)
    {
        case MatrixType_normal:
            return std::make_shared<Matrix>();
        case MatrixType_swapped:
            return std::make_shared<MatrixSwapped>();
        default:
            return nullptr;
    }
}

void MatrixMultiplyTest::prepareTest()
{
    A->fillRandom();
    B->fillRandom();
    kernel = kernelLoader->loadKernel(kernelSourceFilename, kernelName);

    gpuMemory.reserve(3);
    gpuMemory.emplace_back(A->copyToGpu());
    gpuMemory.emplace_back(B->copyToGpu());
    gpuMemory.emplace_back(std::make_shared<ClTypedMemory<float>> (IMatrix::matrixSize*IMatrix::matrixSize));
}

bool MatrixMultiplyTest::performTest()
{
    prepareTest();

    measureTime(kernelName, [&](){(*kernel)[64u][64u](gpuMemory);});

    C = (*A) * (*B);

    gpuMemory[2]->copyOut(C_k->getData(), 0, sizeof(float)*IMatrix::matrixSize*IMatrix::matrixSize);

    if(C == *C_k)
    {
        std::cout << "Correct!" << std::endl;
        return true;
    }
    else
    {
        std::cout << "Something wrong" << std::endl;
        return false;
    }

    return true;
}

