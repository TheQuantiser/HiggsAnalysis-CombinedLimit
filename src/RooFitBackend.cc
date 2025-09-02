#include "RooFitBackend.h"
#include "RooAbsReal.h"

RooFitBackend::RooFitBackend(std::unique_ptr<RooAbsReal> nll) : nll_(std::move(nll)) {}
RooFitBackend::~RooFitBackend() = default;

double RooFitBackend::evaluate() {
    return nll_ ? nll_->getVal() : 0.0;
}

bool RooFitBackend::gradient(std::vector<double>& grad) {
    (void)grad;
    return false;
}

bool RooFitBackend::hessian(std::vector<double>& hess) {
    (void)hess;
    return false;
}
