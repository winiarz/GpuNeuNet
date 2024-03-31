#pragma once

#include "IClIncludePreprocessor.hpp"

class ClIncludePreprocessor : public IClIncludePreprocessor
{
public:
    virtual std::shared_ptr<std::string> replaceIncludes( std::shared_ptr<std::string> source,
                                                          std::set<std::string>& includeDirectories );
private:
    bool isLineIncludeDirective( std::string& line );
    std::string getIncludeFilename( std::string& line );
    std::shared_ptr<std::string> readFile(std::string& filename);
    std::string getIncludeFilePath(std::string&, std::set<std::string>&);
};

