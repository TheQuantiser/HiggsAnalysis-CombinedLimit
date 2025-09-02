#ifndef HIGGSANALYSIS_COMBINEDLIMIT_MOREFITBACKEND_H
#define HIGGSANALYSIS_COMBINEDLIMIT_MOREFITBACKEND_H

#include "HiggsAnalysis/CombinedLimit/interface/ILikelihoodBackend.h"

#ifdef USE_MOREFIT
#include "morefit/include/morefit.hh"
#endif

class MoreFitBackend : public ILikelihoodBackend {
public:
    MoreFitBackend();
    ~MoreFitBackend() override;

    double evaluate() override;
    bool gradient(std::vector<double>& grad) override;
    bool hessian(std::vector<double>& hess) override;

private:
#ifdef USE_MOREFIT
    // placeholders for MoreFit objects
#endif
};

#endif
