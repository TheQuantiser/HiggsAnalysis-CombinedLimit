#include "../interface/CMSInterferenceFunc.h"
#include "TBuffer.h"

// The evaluation of the interference term only needs basic
// vector-matrix operations.  Using Eigen here pulls in a fairly
// heavy dependency, so implement the minimal algebra directly
// using standard containers instead.

class _InterferenceEval {
public:
  _InterferenceEval(std::vector<std::vector<double>> scaling_in, size_t ncoef)
      : ncoef_(ncoef), binscaling_(std::move(scaling_in)), coefficients_(ncoef, 0.), values_(binscaling_.size()) {}

  inline void setCoefficient(size_t i, double val) { coefficients_[i] = val; }

  void computeValues() {
    for (size_t ibin = 0; ibin < binscaling_.size(); ++ibin) {
      const auto& mat = binscaling_[ibin];
      double result = 0.;
      size_t k = 0;
      for (size_t r = 0; r < ncoef_; ++r) {
        for (size_t c = 0; c <= r; ++c) {
          double m = mat[k++];
          double term = m * coefficients_[r] * coefficients_[c];
          result += (r == c) ? term : 2. * term;
        }
      }
      values_[ibin] = result;
    }
  }

  const std::vector<double>& getValues() const { return values_; }

private:
  size_t ncoef_;
  std::vector<std::vector<double>> binscaling_;
  std::vector<double> coefficients_;
  std::vector<double> values_;
};

CMSInterferenceFunc::CMSInterferenceFunc() {};

CMSInterferenceFunc::CMSInterferenceFunc(CMSInterferenceFunc const& other, const char* name)
    : CMSExternalMorph(other, name),
      coefficients_("coefficients", this, other.coefficients_),
      binscaling_(other.binscaling_),
      sentry_(name ? TString(name) + "_sentry" : TString(other.GetName()) + "_sentry", "") {}

CMSInterferenceFunc::CMSInterferenceFunc(const char* name,
                                         const char* title,
                                         RooRealVar& x,
                                         const std::vector<double>& edges,
                                         RooArgList const& coefficients,
                                         const std::vector<std::vector<double>> binscaling)
    : CMSExternalMorph(name, title, x, edges),
      coefficients_("coefficients", "", this),
      binscaling_(binscaling),
      sentry_(TString(name) + "_sentry", "") {
  coefficients_.add(coefficients);
}

CMSInterferenceFunc::~CMSInterferenceFunc() = default;

void CMSInterferenceFunc::printMultiline(std::ostream& os, Int_t contents, Bool_t verbose, TString indent) const {
  RooAbsReal::printMultiline(os, contents, verbose, indent);
  os << ">> Sentry: " << (sentry_.good() ? "clean" : "dirty") << "\n";
  sentry_.Print("v");
}

void CMSInterferenceFunc::initialize() const {
  // take the opportunity to validate persistent data
  size_t nbins = edges_.size() - 1;
  size_t ncoef = coefficients_.getSize();

  for (size_t i = 0; i < ncoef; ++i) {
    if (coefficients_.at(i) == nullptr) {
      throw std::invalid_argument("Lost coefficient " + std::to_string(i));
    }
    if (not coefficients_.at(i)->InheritsFrom("RooAbsReal")) {
      throw std::invalid_argument("Coefficient " + std::to_string(i) + " is not a RooAbsReal");
    }
  }
  if (binscaling_.size() != nbins) {
    throw std::invalid_argument("Number of bins as determined from bin edges (" + std::to_string(nbins) +
                                ") does not match bin"
                                " scaling array (" +
                                std::to_string(binscaling_.size()) + ")");
  }
  for (size_t i = 0; i < nbins; ++i) {
    if (binscaling_[i].size() != ncoef * (ncoef + 1) / 2) {
      throw std::invalid_argument("Length of bin scaling matrix lower triangle for bin " + std::to_string(i) + " (" +
                                  std::to_string(binscaling_[i].size()) + ") is not consistent" +
                                  " with the length of the coefficient array (" + std::to_string(ncoef) + ")");
    }
  }

  sentry_.SetName(TString(GetName()) + "_sentry");
  sentry_.addVars(coefficients_);
  sentry_.setValueDirty();
  evaluator_ = std::make_unique<_InterferenceEval>(binscaling_, ncoef);
}

void CMSInterferenceFunc::updateCache() const {
  for (int i = 0; i < coefficients_.getSize(); ++i) {
    auto* coef = static_cast<RooAbsReal const*>(coefficients_.at(i));
    if (coef == nullptr)
      throw std::runtime_error("Lost coef!");
    evaluator_->setCoefficient(i, coef->getVal());
  }
  evaluator_->computeValues();
  sentry_.reset();
}

const std::vector<double>& CMSInterferenceFunc::batchGetBinValues() const {
  if (not evaluator_)
    initialize();
  if (not sentry_.good())
    updateCache();
  return evaluator_->getValues();
}
