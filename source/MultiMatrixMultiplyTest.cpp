#include "MultiMatrixMultiplyTest.hpp"
#include "time.hpp"
#include "ClKernel.hpp"

std::shared_ptr<ClKernelFromSourceLoader> MultiMatrixMultiplyTest::kernelLoader = nullptr;

MultiMatrixMultiplyTest::MultiMatrixMultiplyTest(MultiMatrixType typeLeft,
                                                 MatrixType typeRight,
                                                 MultiMatrixType typeResult,
                                                 std::string p_fileName,
                                                 std::string p_kernelName) :
            fileName(p_fileName), kernelName(p_kernelName)
{
    A = createMultiMatrixOfType(typeLeft);
    B = createMatrixOfType(typeRight);
    C = createMultiMatrixOfType(typeResult);
    C_k = createMultiMatrixOfType(typeResult);

    if(kernelLoader == nullptr)
    {
        std::set<std::string> clIncludeDirs;
        kernelLoader = std::make_shared<ClKernelFromSourceLoader>(clIncludeDirs);
    }
}

std::shared_ptr<IMatrix> MultiMatrixMultiplyTest::createMatrixOfType(MatrixType matrixType)
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

std::shared_ptr<IMultiMatrix> MultiMatrixMultiplyTest::createMultiMatrixOfType(MultiMatrixType multiMatrixType)
{
    switch(multiMatrixType)
    {
      case MultiMatrixType_separateNormal:
        return std::make_shared<MultiMatrix_SeparateNormal>();
    }
    return nullptr;
}

void MultiMatrixMultiplyTest::prepareTest()
{
    A->fillRandom();
    B->fillRandom();
    kernel = kernelLoader->loadKernel(fileName, kernelName);

    gpuMemory.reserve(3);
    gpuMemory.emplace_back(A->copyToGpu());
    gpuMemory.emplace_back(B->copyToGpu());
    gpuMemory.emplace_back(std::make_shared<ClTypedMemory<float>> (IMultiMatrix::matrixSize*IMultiMatrix::matrixSize*IMultiMatrix::matrixCount));

}

bool MultiMatrixMultiplyTest::performTest()
{
  prepareTest();
  C = A * B;

  measureTime(kernelName, [&](){(*kernel)[1u][256u](gpuMemory);});
  gpuMemory[2]->copyOut(C_k->getData(), 0, sizeof(float)*IMultiMatrix::matrixSize*IMultiMatrix::matrixSize*IMultiMatrix::matrixCount);

  if(*C == *C_k)
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
