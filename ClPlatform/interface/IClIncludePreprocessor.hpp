#pragma once

#include <memory>
#include "stl.hpp"

class IClIncludePreprocessor
{
public:
    virtual std::shared_ptr<std::string> replaceIncludes( std::shared_ptr<std::string> source,
                                                          std::set<std::string>& includeDirectories ) = 0;
};

