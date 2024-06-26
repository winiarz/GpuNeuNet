#include "time.hpp"
#include <iostream>
#include <chrono>

void measureTime(std::string description, std::function<void()> operationToMeasure)
{
    auto start = std::chrono::high_resolution_clock::now();
    operationToMeasure();
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    std::cout << "Operation: '" << description << "' has taken " << (microseconds/1000) << "." << (microseconds/100%10) << (microseconds/10%10) << (microseconds%10) << " miliseconds" << std::endl;
}

