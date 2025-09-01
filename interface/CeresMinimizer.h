#ifndef HiggsAnalysis_CombinedLimit_CeresMinimizer_h
#define HiggsAnalysis_CombinedLimit_CeresMinimizer_h

#include <Math/Minimizer.h>
#if __has_include(<Math/IGradientFunctionMultiDim.h>)
#include <Math/IGradientFunctionMultiDim.h>
using RootIMultiGradFunction = ROOT::Math::IGradientFunctionMultiDim;
#else
#include "Math/IFunction.h"
using RootIMultiGradFunction = ROOT::Math::IMultiGradFunction;
#endif
#include <ceres/ceres.h>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <RVersion.h>

/// Minimizer interface using Ceres Solver
class CeresMinimizer : public ROOT::Math::Minimizer {
public:
  CeresMinimizer(const char *name = nullptr);
  ~CeresMinimizer() override;

#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 32, 0)
// ROOT ≥ 6.32: ROOT::Math::Minimizer has no Name(), and
// ProvidesGradient/ProvidesHessian are not virtual in the base.
// -> Do not declare Name(); keep helpers as non-virtuals.
  bool ProvidesGradient() const { return true; }
  bool ProvidesHessian() const { return true; }

#elif ROOT_VERSION_CODE >= ROOT_VERSION(6, 30, 0)
// ROOT 6.30.x: some builds changed/experimented with Name()'s return type.
// Provide a non-virtual Name() to keep compatibility without risking a bad override.
# if defined(__cpp_lib_string_view) && __cpp_lib_string_view >= 201606
  std::string_view Name() const { return "Ceres"; }
# else
  std::string Name() const { return "Ceres"; }
# endif
  bool ProvidesGradient() const { return true; }
  bool ProvidesHessian() const { return true; }

#elif ROOT_VERSION_CODE >= ROOT_VERSION(6, 24, 0)
// Older 6.24–6.28 era: Name() was const char*. We don't need 'override' here;
// it will still override if the signature matches that ROOT version.
  const char *Name() const { return "Ceres"; }
  bool ProvidesGradient() const { return true; }
  bool ProvidesHessian() const { return true; }

#else
// Very old ROOT
  const char *Name() const { return "Ceres"; }
  bool ProvidesGradient() const { return true; }
  bool ProvidesHessian() const { return true; }
#endif

  void Clear() override;
  void SetFunction(const ROOT::Math::IMultiGenFunction &func) override;

  bool SetVariable(unsigned int ivar, const std::string &name, double val, double step) override;
  bool SetLimitedVariable(
      unsigned int ivar, const std::string &name, double val, double step, double lower, double upper) override;
  bool SetFixedVariable(unsigned int ivar, const std::string &name, double val) override;

  bool Minimize() override;

  double MinValue() const override { return fMinVal_; }
  double Edm() const override { return edm_; }

  const double *X() const override { return x_.data(); }
  const double *MinGradient() const override { return grad_.empty() ? nullptr : grad_.data(); }
  unsigned int NCalls() const override { return nCalls_; }
  unsigned int NDim() const override { return nDim_; }
  unsigned int NFree() const override { return nFree_; }

  // ROOT's Minimizer did not historically expose a virtual Status()
  // method, so we avoid using the 'override' keyword here to keep
  // compatibility across ROOT versions while still allowing callers to
  // query the minimizer termination code when available.
  int Status() const { return status_; }

  bool ProvidesError() const override { return true; }
  const double *Errors() const override { return errors_.empty() ? nullptr : errors_.data(); }
  double CovMatrix(unsigned int i, unsigned int j) const override {
    if (cov_.empty() || i >= nDim_ || j >= nDim_)
      return 0.0;
    return cov_[i * nDim_ + j];
  }

  void Gradient(const double *x, double *grad) const;
  void Hessian(const double *x, double *hes) const;

private:
  void ComputeGradientAndHessian(const double *x);

  struct CostFunction : public ceres::CostFunction {
    CostFunction(const RootIMultiGradFunction *f);
    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;
    const RootIMultiGradFunction *func;
  };

  const ROOT::Math::IMultiGenFunction *func_;
  const RootIMultiGradFunction *gradFunc_;
  unsigned int nDim_;
  unsigned int nFree_;
  unsigned int nCalls_;

  std::vector<double> x_;
  std::vector<double> step_;
  std::vector<double> lower_;
  std::vector<double> upper_;
  std::vector<bool> isFixed_;

  std::vector<double> grad_;
  std::vector<double> hess_;
  std::vector<double> cov_;
  std::vector<double> errors_;

  double fMinVal_;
  double edm_;

  double numDiffStep_;
  bool forceNumeric_;

  int status_;
};

#endif
