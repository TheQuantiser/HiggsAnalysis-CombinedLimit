#include <boost/program_options.hpp>
#include "../../interface/CascadeMinimizer.h"
#include <iostream>

int main() {
    using namespace boost::program_options;
    variables_map vm;
    vm.insert(std::make_pair("cminDefaultMinimizerType", variable_value(std::string("Ceres"), false)));
    try {
        CascadeMinimizer::applyOptions(vm);
    } catch (const std::runtime_error &e) {
        std::cout << "Caught expected error: " << e.what() << std::endl;
        return 0;
    }
    std::cerr << "Expected failure to load Ceres minimizer plugin" << std::endl;
    return 1;
}
