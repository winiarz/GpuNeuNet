#include "ClKernelFromSourceLoader.hpp"
#include "ClPlatform.hpp"
#include "ClKernel.hpp"
#include "ClError.hpp"
#include "stl.hpp"

ClKernelFromSourceLoader::ClKernelFromSourceLoader(std::set<std::string> p_includeDirectories) :
    includeDirectories(p_includeDirectories)
{
}

std::shared_ptr<ClKernel> ClKernelFromSourceLoader::loadKernel(std::string filename, std::string kernelName)
{
    std::shared_ptr<std::string> source = readFile(filename);

    std::shared_ptr<std::string> sourceWithReplacedIncludes = includePreprocessor.replaceIncludes(source, includeDirectories);
    cl_program program = compileSource( sourceWithReplacedIncludes );
    return std::make_shared<ClKernel>(program, kernelName.c_str());
}

std::shared_ptr<ClKernel> ClKernelFromSourceLoader::loadKernel(std::string filename)
{

    std::shared_ptr<std::string> source = readFile(filename);

    std::shared_ptr<std::string> sourceWithReplacedIncludes = includePreprocessor.replaceIncludes(source, includeDirectories);
    cl_program program = compileSource( sourceWithReplacedIncludes );
    return std::make_shared<ClKernel>(program);
}

cl_program ClKernelFromSourceLoader::compileSource(std::shared_ptr<std::string> source)
{
    ClPlatform& platform = ClPlatform::getPlatform();

    cl_int error;

    const char* sourceCString = source->c_str();

    cl_program program = clCreateProgramWithSource(platform.getContext(),
                                                   1,
                                                   (const char**)&sourceCString,
                                                   NULL,
                                                   &error);

    if ( (program == NULL) || ( error != CL_SUCCESS ) )
    {
        std::cerr << "Failed to create CL program from source, error = " << error << std::endl;
        throw OPEN_CL_ERROR;
    }

    error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (error != CL_SUCCESS)
    {
        char buildLog[MAX_BUILD_LOG_SIZE];
        clGetProgramBuildInfo(program,
                              platform.getDevice(),
                              CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog),
                              buildLog,
                              NULL);

        std::cout << "Error in kernel: ";
        std::cout << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

std::shared_ptr<std::string> ClKernelFromSourceLoader::readFile(std::string& filename)
{
    std::ifstream file(filename.c_str());
    if ( !file.is_open() )
    {
        std::cerr << "error while reading file " << filename << std::endl;
        throw FILE_READ_ERROR;
    }

    std::shared_ptr<std::string> text = std::make_shared<std::string>();

    file.seekg(0, std::ios::end);
    text->reserve(file.tellg());
    file.seekg(0, std::ios::beg);

    text->assign( std::istreambuf_iterator<char>(file),
                  std::istreambuf_iterator<char>() );

    return text;
}

ClKernelFromSourceLoader::~ClKernelFromSourceLoader()
{

}

