#ifndef HIGGSANALYSIS_COMBINEDLIMIT_ILIKELIHOODBACKEND_H
#define HIGGSANALYSIS_COMBINEDLIMIT_ILIKELIHOODBACKEND_H

#include <vector>

class RooAbsData;
class RooArgList;

class ILikelihoodBackend {
public:
    virtual ~ILikelihoodBackend() = default;
    /// Called when a new dataset is bound to the likelihood
    virtual void prepareForDataset(const RooAbsData* data) { (void)data; }
    /// Called when parameter pointers change
    virtual void prepareForParams(const RooArgList* params) { (void)params; }
    /// Evaluate the negative log-likelihood
    virtual double evaluate() = 0;
    /// Fill gradient vector; return false if not available
    virtual bool gradient(std::vector<double>& grad) { (void)grad; return false; }
    /// Fill Hessian matrix (row-major); return false if not available
    virtual bool hessian(std::vector<double>& hess) { (void)hess; return false; }
};

#endif
