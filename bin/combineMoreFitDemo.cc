#include "../src/MoreFitBackend.h"
#include <iostream>

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    MoreFitBackend backend;
    std::cout << "MoreFit demo evaluate = " << backend.evaluate() << std::endl;
    return 0;
}
