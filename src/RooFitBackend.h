#ifndef HIGGSANALYSIS_COMBINEDLIMIT_ROOFITBACKEND_H
#define HIGGSANALYSIS_COMBINEDLIMIT_ROOFITBACKEND_H

#include "HiggsAnalysis/CombinedLimit/interface/ILikelihoodBackend.h"
#include <memory>

class RooAbsReal;

class RooFitBackend : public ILikelihoodBackend {
public:
    explicit RooFitBackend(std::unique_ptr<RooAbsReal> nll);
    ~RooFitBackend() override;

    double evaluate() override;
    bool gradient(std::vector<double>& grad) override;
    bool hessian(std::vector<double>& hess) override;

private:
    std::unique_ptr<RooAbsReal> nll_;
};

#endif
