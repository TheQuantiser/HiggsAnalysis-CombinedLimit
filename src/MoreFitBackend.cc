#include "MoreFitBackend.h"

MoreFitBackend::MoreFitBackend() = default;
MoreFitBackend::~MoreFitBackend() = default;

double MoreFitBackend::evaluate() {
#ifdef USE_MOREFIT
    // real implementation will call morefit graph execution
    return 0.0;
#else
    return 0.0;
#endif
}

bool MoreFitBackend::gradient(std::vector<double>& grad) {
    (void)grad;
#ifdef USE_MOREFIT
    return false;
#else
    return false;
#endif
}

bool MoreFitBackend::hessian(std::vector<double>& hess) {
    (void)hess;
#ifdef USE_MOREFIT
    return false;
#else
    return false;
#endif
}
