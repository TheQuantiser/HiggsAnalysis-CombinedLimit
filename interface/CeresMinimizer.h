#ifndef HiggsAnalysis_CombinedLimit_CeresMinimizer_h
#define HiggsAnalysis_CombinedLimit_CeresMinimizer_h

#include <Math/Minimizer.h>
#include <Math/IMultiGradFunction.h>
#include <ceres/ceres.h>
#include <string>
#include <vector>
#include <memory>

/// Minimizer interface using Ceres Solver
class CeresMinimizer : public ROOT::Math::Minimizer {
public:
    CeresMinimizer(const char *name = nullptr);
    ~CeresMinimizer() override;

    const char * Name() const override { return "Ceres"; }

    void Clear() override;
    void SetFunction(const ROOT::Math::IMultiGenFunction & func) override;

    bool SetVariable(unsigned int ivar, const std::string & name, double val, double step) override;
    bool SetLimitedVariable(unsigned int ivar, const std::string & name, double val, double step, double lower, double upper) override;
    bool SetFixedVariable(unsigned int ivar, const std::string & name, double val) override;

    bool Minimize() override;

    double MinValue() const override { return fMinVal_; }
    double Edm() const override { return edm_; }

    const double * X() const override { return x_.data(); }
    const double * MinGradient() const override { return grad_.empty() ? nullptr : grad_.data(); }
    unsigned int NCalls() const override { return nCalls_; }
    unsigned int NDim() const override { return nDim_; }
    unsigned int NFree() const override { return nFree_; }

    bool ProvidesError() const override { return true; }
    const double * Errors() const override { return errors_.empty() ? nullptr : errors_.data(); }
    double CovMatrix(unsigned int i, unsigned int j) const override {
        if (cov_.empty() || i >= nDim_ || j >= nDim_) return 0.0;
        return cov_[i*nDim_ + j];
    }

    bool ProvidesGradient() const override { return true; }
    bool ProvidesHessian() const override { return true; }

    void Gradient(const double *x, double *grad) const;
    void Hessian(const double *x, double *hes) const;

private:
    void ComputeGradientAndHessian(const double *x);

    struct CostFunction : public ceres::CostFunction {
        CostFunction(const ROOT::Math::IMultiGradFunction *f);
        bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const override;
        const ROOT::Math::IMultiGradFunction *func;
    };

    const ROOT::Math::IMultiGenFunction *func_;
    const ROOT::Math::IMultiGradFunction *gradFunc_;
    unsigned int nDim_;
    unsigned int nFree_;
    unsigned int nCalls_;

    std::vector<double> x_;
    std::vector<double> step_;
    std::vector<double> lower_;
    std::vector<double> upper_;
    std::vector<bool>   isFixed_;

    std::vector<double> grad_;
    std::vector<double> hess_;
    std::vector<double> cov_;
    std::vector<double> errors_;

    double fMinVal_;
    double edm_;

    double numDiffStep_;
    bool forceNumeric_;
};

#endif
