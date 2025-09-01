#include "../interface/CeresMinimizer.h"
#include "../interface/CombineLogger.h"
#include <Math/MinimizerOptions.h>
#include <TPluginManager.h>
#include <TMatrixDSym.h>
#include <ceres/loss_function.h>
#include <ceres/covariance.h>
#include <TString.h>
#include <fstream>
#include <cstdlib>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>
#include <thread>
#include <chrono>
#include <memory>
#include <utility>

CeresMinimizer::CeresMinimizer(const char *)
    : func_(nullptr),
      gradFunc_(nullptr),
      nDim_(0),
      nFree_(0),
      nCalls_(0),
      fMinVal_(0.0),
      edm_(0.0),
      numDiffStep_(1e-4),
      forceNumeric_(false) {}

CeresMinimizer::~CeresMinimizer() {}

void CeresMinimizer::Clear() {
  func_ = nullptr;
  gradFunc_ = nullptr;
  nDim_ = nFree_ = 0;
  nCalls_ = 0;
  x_.clear();
  step_.clear();
  lower_.clear();
  upper_.clear();
  isFixed_.clear();
  grad_.clear();
  hess_.clear();
  cov_.clear();
  errors_.clear();
  fMinVal_ = 0.0;
  edm_ = 0.0;
  numDiffStep_ = 1e-4;
  forceNumeric_ = false;
}

void CeresMinimizer::SetFunction(const ROOT::Math::IMultiGenFunction &func) {
  func_ = &func;
  gradFunc_ = dynamic_cast<const RootIMultiGradFunction *>(func_);
  nDim_ = func.NDim();
  nFree_ = nDim_;
  x_.assign(nDim_, 0.0);
  step_.assign(nDim_, 0.1);
  lower_.assign(nDim_, -std::numeric_limits<double>::infinity());
  upper_.assign(nDim_, std::numeric_limits<double>::infinity());
  isFixed_.assign(nDim_, false);
}

bool CeresMinimizer::SetVariable(unsigned int i, const std::string &, double val, double step) {
  if (i >= nDim_)
    return false;
  x_[i] = val;
  step_[i] = step;
  isFixed_[i] = false;
  return true;
}

bool CeresMinimizer::SetLimitedVariable(
    unsigned int i, const std::string &, double val, double step, double lower, double upper) {
  if (i >= nDim_)
    return false;
  x_[i] = val;
  step_[i] = step;
  lower_[i] = lower;
  upper_[i] = upper;
  isFixed_[i] = false;
  return true;
}

bool CeresMinimizer::SetFixedVariable(unsigned int i, const std::string &, double val) {
  if (i >= nDim_)
    return false;
  x_[i] = val;
  isFixed_[i] = true;
  nFree_--;
  return true;
}

CeresMinimizer::CostFunction::CostFunction(const RootIMultiGradFunction *f) : func(f) {
  set_num_residuals(1);
  mutable_parameter_block_sizes()->push_back(f->NDim());
}

bool CeresMinimizer::CostFunction::Evaluate(double const *const *parameters,
                                            double *residuals,
                                            double **jacobians) const {
  const double *x = parameters[0];
  double fval = (*func)(x);
  double safeFval = std::max(fval, 0.0);
  double sqrtFval = std::sqrt(safeFval);
  residuals[0] = sqrtFval;
  if (jacobians && jacobians[0]) {
    std::vector<double> grad(func->NDim());
    func->Gradient(x, &grad[0]);
    double coeff = sqrtFval > 0 ? 0.5 / sqrtFval : 0.0;
    for (unsigned int i = 0; i < func->NDim(); ++i)
      jacobians[0][i] = coeff * grad[i];
  }
  return true;
}

struct NumericCostFunction : public ceres::CostFunction {
  NumericCostFunction(const ROOT::Math::IMultiGenFunction *f, double step, bool central)
      : func(f), step(step), useCentral(central) {
    set_num_residuals(1);
    mutable_parameter_block_sizes()->push_back(f->NDim());
  }
  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
    const double *x = parameters[0];
    double fval = (*func)(x);
    double safeFval = std::max(fval, 0.0);
    double sqrtFval = std::sqrt(safeFval);
    residuals[0] = sqrtFval;
    if (jacobians && jacobians[0]) {
      std::vector<double> xtmp(func->NDim());
      std::copy(x, x + func->NDim(), xtmp.begin());
      double coeff = sqrtFval > 0 ? 0.5 / sqrtFval : 0.0;
      for (unsigned int i = 0; i < func->NDim(); ++i) {
        if (useCentral) {
          xtmp[i] += step;
          double fp = (*func)(xtmp.data());
          xtmp[i] -= 2 * step;
          double fm = (*func)(xtmp.data());
          jacobians[0][i] = coeff * (fp - fm) / (2 * step);
          xtmp[i] = x[i];
        } else {
          xtmp[i] += step;
          double fp = (*func)(xtmp.data());
          jacobians[0][i] = coeff * (fp - fval) / step;
          xtmp[i] = x[i];
        }
      }
    }
    return true;
  }
  const ROOT::Math::IMultiGenFunction *func;
  double step;
  bool useCentral;
};

bool CeresMinimizer::Minimize() {
  if (!func_)
    return false;

  // retrieve solver configuration from environment
  int maxIter = 1000;
  if (const char *env = std::getenv("CERES_MAX_ITERATIONS"))
    maxIter = std::atoi(env);
  std::string linearSolver =
      std::getenv("CERES_LINEAR_SOLVER") ? std::getenv("CERES_LINEAR_SOLVER") : std::string("dense_qr");
  unsigned int multiStart = 1;
  if (const char *env = std::getenv("CERES_MULTI_START"))
    multiStart = std::max(1, std::atoi(env));
  double jitter = 1.0;
  if (const char *env = std::getenv("CERES_JITTER"))
    jitter = std::max(0.0, std::atof(env));
  std::string jitterDist = std::getenv("CERES_JITTER_DIST") ? std::getenv("CERES_JITTER_DIST") : std::string("uniform");
  bool verbose = std::getenv("CERES_VERBOSE") != nullptr;
  bool progress = std::getenv("CERES_PROGRESS") != nullptr;
  int numThreads = 1;
  if (const char *env = std::getenv("CERES_NUM_THREADS"))
    numThreads = std::max(1, std::atoi(env));
  else if (std::getenv("CERES_AUTO_THREADS")) {
    unsigned hw = std::thread::hardware_concurrency();
    numThreads = hw ? static_cast<int>(hw) : 1;
  }
  unsigned int seed = 12345;
  if (const char *env = std::getenv("CERES_RANDOM_SEED"))
    seed = static_cast<unsigned int>(std::strtoul(env, nullptr, 10));
  bool forceNumeric = std::getenv("CERES_FORCE_NUMERIC") != nullptr;
  forceNumeric_ = forceNumeric;

  double funTol = ROOT::Math::MinimizerOptions::DefaultTolerance();
  if (const char *env = std::getenv("CERES_FUNCTION_TOLERANCE"))
    funTol = std::atof(env);
  double gradTol = funTol;
  if (const char *env = std::getenv("CERES_GRADIENT_TOLERANCE"))
    gradTol = std::atof(env);
  double parTol = funTol;
  if (const char *env = std::getenv("CERES_PARAMETER_TOLERANCE"))
    parTol = std::atof(env);
  std::string algo = ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo();
  if (const char *env = std::getenv("CERES_ALGO"))
    algo = env;
  double numericStep = 1e-4;
  if (const char *env = std::getenv("CERES_NUMERIC_DIFF_STEP"))
    numericStep = std::abs(std::atof(env));
  if (numericStep <= 0)
    numericStep = 1e-4;
  numDiffStep_ = numericStep;
  std::string diffMethod = std::getenv("CERES_DIFF_METHOD") ? std::getenv("CERES_DIFF_METHOD") : std::string("central");
  bool centralDiff = (diffMethod != "forward");
  std::string lossStr = std::getenv("CERES_LOSS_FUNCTION") ? std::getenv("CERES_LOSS_FUNCTION") : std::string("none");
  double initRadius = 1.0;
  if (const char *env = std::getenv("CERES_INITIAL_RADIUS"))
    initRadius = std::max(1e-12, std::atof(env));
  double lossScale = 1.0;
  if (const char *env = std::getenv("CERES_LOSS_SCALE"))
    lossScale = std::max(0.0, std::atof(env));
  std::string logFile = std::getenv("CERES_LOG_FILE") ? std::getenv("CERES_LOG_FILE") : std::string();
  double maxTime = 0.0;
  if (const char *env = std::getenv("CERES_MAX_TIME"))
    maxTime = std::max(0.0, std::atof(env));
  double boundRelax = 0.0;
  if (const char *env = std::getenv("CERES_BOUND_RELAX"))
    boundRelax = std::max(0.0, std::atof(env));

  std::vector<double> xbest = x_, xinit = x_;
  double bestFval = std::numeric_limits<double>::infinity();
  ceres::Solver::Summary bestSummary;
  std::unique_ptr<ceres::Problem> bestProblem;

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::normal_distribution<double> gdist(0.0, 1.0);
  bool useGauss = (jitterDist == "gaussian");
  auto startTime = std::chrono::steady_clock::now();

  unsigned int it = 0;
  for (; it < multiStart; ++it) {
    if (maxTime > 0.0) {
      double elapsed =
          std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - startTime).count();
      if (elapsed >= maxTime)
        break;
    }
    if (it > 0) {
      for (unsigned int i = 0; i < nDim_; ++i) {
        double r = useGauss ? gdist(rng) : dist(rng);
        x_[i] = xinit[i] + r * step_[i] * jitter;
      }
    } else {
      x_ = xinit;
    }

    auto problem = std::make_unique<ceres::Problem>();
    ceres::CostFunction *cost = nullptr;
    if (gradFunc_ && !forceNumeric)
      cost = new CostFunction(gradFunc_);
    else
      cost = new NumericCostFunction(func_, numericStep, centralDiff);
    ceres::LossFunction *loss = nullptr;
    if (lossStr == "huber")
      loss = new ceres::HuberLoss(lossScale);
    else if (lossStr == "cauchy")
      loss = new ceres::CauchyLoss(lossScale);
    problem->AddResidualBlock(cost, loss, x_.data());
    for (unsigned int i = 0; i < nDim_; ++i) {
      if (isFixed_[i])
        problem->SetParameterBlockConstant(&x_[i]);
      else {
        if (std::isfinite(lower_[i]))
          problem->SetParameterLowerBound(x_.data(), i, lower_[i] - boundRelax);
        if (std::isfinite(upper_[i]))
          problem->SetParameterUpperBound(x_.data(), i, upper_[i] + boundRelax);
      }
    }
    ceres::Solver::Options options;
    options.max_num_iterations = maxIter;
    options.function_tolerance = funTol;
    options.gradient_tolerance = gradTol;
    options.parameter_tolerance = parTol;
    options.minimizer_type = (algo == "LineSearch" ? ceres::LINE_SEARCH : ceres::TRUST_REGION);
    if (linearSolver == "dense_qr")
      options.linear_solver_type = ceres::DENSE_QR;
    else if (linearSolver == "dense_normal_cholesky")
      options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    else if (linearSolver == "iterative_schur")
      options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    else if (linearSolver == "sparse_normal_cholesky")
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    else if (linearSolver == "dense_schur")
      options.linear_solver_type = ceres::DENSE_SCHUR;
    else if (linearSolver == "sparse_schur")
      options.linear_solver_type = ceres::SPARSE_SCHUR;
    else {
      CombineLogger::instance().log("CeresMinimizer.cc",
                                    __LINE__,
                                    Form("Unknown linear solver %s, using dense_qr", linearSolver.c_str()),
                                    __func__);
      options.linear_solver_type = ceres::DENSE_QR;
    }
    options.num_threads = numThreads;
    options.initial_trust_region_radius = initRadius;
    options.minimizer_progress_to_stdout = progress;
    ceres::Solver::Summary summary;
    ceres::Solve(options, problem.get(), &summary);
    if (!summary.IsSolutionUsable()) {
      ceres::Solver::Options altOptions = options;
      altOptions.minimizer_type =
          (options.minimizer_type == ceres::LINE_SEARCH ? ceres::TRUST_REGION : ceres::LINE_SEARCH);
      ceres::Solver::Summary altSummary;
      ceres::Solve(altOptions, problem.get(), &altSummary);
      if (altSummary.IsSolutionUsable()) {
        if (verbose)
          CombineLogger::instance().log(
              "CeresMinimizer.cc", __LINE__, "retry with alternate algorithm succeeded", __func__);
        summary = altSummary;
      }
    }
    double fval = (*func_)(x_.data());
    if (verbose)
      CombineLogger::instance().log(
          "CeresMinimizer.cc", __LINE__, Form("multi-start %u fval %.6f", it, fval), __func__);
    if (summary.IsSolutionUsable() && fval < bestFval) {
      bestFval = fval;
      xbest = x_;
      bestSummary = summary;
      bestProblem = std::move(problem);
    }
  }
  if (maxTime > 0.0) {
    double elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - startTime).count();
    if (elapsed >= maxTime)
      CombineLogger::instance().log(
          "CeresMinimizer.cc", __LINE__, "time limit reached before completing all starts", __func__);
  }

  x_ = xbest;
  nCalls_ = bestSummary.num_successful_steps + bestSummary.num_unsuccessful_steps;
  fMinVal_ = bestFval;
  grad_.assign(nDim_, 0.0);
  hess_.assign(nDim_ * nDim_, 0.0);
  cov_.assign(nDim_ * nDim_, 0.0);
  errors_.assign(nDim_, 0.0);

  Gradient(x_.data(), grad_.data());

  bool covOK = false;
  if (bestProblem && bestSummary.IsSolutionUsable()) {
    ceres::Covariance::Options covOpts;
    ceres::Covariance covariance(covOpts);
    std::vector<std::pair<const double *, const double *>> blocks;
    blocks.emplace_back(x_.data(), x_.data());
    if (covariance.Compute(blocks, bestProblem.get())) {
      covariance.GetCovarianceBlock(x_.data(), x_.data(), &cov_[0]);
      covOK = true;
      TMatrixDSym covmat(nDim_);
      for (unsigned int i = 0; i < nDim_; ++i)
        for (unsigned int j = 0; j < nDim_; ++j)
          covmat(i, j) = cov_[i * nDim_ + j];
      TMatrixDSym hmat(covmat);
      hmat.Invert();
      for (unsigned int i = 0; i < nDim_; ++i) {
        for (unsigned int j = 0; j < nDim_; ++j) {
          hess_[i * nDim_ + j] = hmat(i, j);
        }
        errors_[i] = covmat(i, i) > 0 ? std::sqrt(covmat(i, i)) : 0.0;
      }
    } else {
      CombineLogger::instance().log(
          "CeresMinimizer.cc", __LINE__, "covariance computation failed, using numeric Hessian", __func__);
    }
  }

  if (!covOK) {
    ComputeGradientAndHessian(x_.data());
    if (!hess_.empty()) {
      TMatrixDSym hmat(nDim_);
      for (unsigned int i = 0; i < nDim_; ++i)
        for (unsigned int j = 0; j < nDim_; ++j)
          hmat(i, j) = hess_[i * nDim_ + j];
      hmat.Invert();
      for (unsigned int i = 0; i < nDim_; ++i) {
        for (unsigned int j = 0; j < nDim_; ++j) {
          cov_[i * nDim_ + j] = hmat(i, j);
        }
        errors_[i] = hmat(i, i) > 0 ? std::sqrt(hmat(i, i)) : 0.0;
      }
    }
  }
  edm_ = 0.0;
  for (unsigned int i = 0; i < nDim_; ++i)
    for (unsigned int j = 0; j < nDim_; ++j)
      edm_ += 0.5 * grad_[i] * cov_[i * nDim_ + j] * grad_[j];

  CombineLogger::instance().log("CeresMinimizer.cc", __LINE__, bestSummary.BriefReport(), __func__);
  if (verbose)
    CombineLogger::instance().log("CeresMinimizer.cc", __LINE__, bestSummary.FullReport(), __func__);
  if (!logFile.empty()) {
    std::ofstream ofs(logFile.c_str(), std::ios::app);
    if (ofs)
      ofs << bestSummary.FullReport() << std::endl;
  }

  if (multiStart > 1 && jitter == 0.0)
    CombineLogger::instance().log(
        "CeresMinimizer.cc", __LINE__, "multi-start requested without jitter; results may be identical", __func__);

  return bestSummary.IsSolutionUsable();
}

void CeresMinimizer::Gradient(const double *x, double *grad) const {
  if (gradFunc_ && !forceNumeric_) {
    gradFunc_->Gradient(x, grad);
  } else {
    std::vector<double> xtmp(nDim_);
    std::copy(x, x + nDim_, xtmp.begin());
    for (unsigned int i = 0; i < nDim_; ++i) {
      double step = step_[i] * numDiffStep_;
      if (step == 0)
        step = numDiffStep_;
      xtmp[i] += step;
      double fp = (*func_)(xtmp.data());
      xtmp[i] -= 2 * step;
      double fm = (*func_)(xtmp.data());
      grad[i] = (fp - fm) / (2 * step);
      xtmp[i] = x[i];
    }
  }
}

void CeresMinimizer::Hessian(const double *x, double *hes) const {
  std::vector<double> xtmp(nDim_);
  std::copy(x, x + nDim_, xtmp.begin());
  std::vector<double> grad1(nDim_), grad2(nDim_);
  Gradient(x, &grad1[0]);
  for (unsigned int j = 0; j < nDim_; ++j) {
    double step = step_[j] * numDiffStep_;
    if (step == 0)
      step = numDiffStep_;
    xtmp[j] += step;
    Gradient(xtmp.data(), &grad2[0]);
    for (unsigned int i = 0; i < nDim_; ++i) {
      hes[i * nDim_ + j] = (grad2[i] - grad1[i]) / step;
    }
    xtmp[j] = x[j];
  }
}

void CeresMinimizer::ComputeGradientAndHessian(const double *x) {
  std::vector<double> xtmp(nDim_);
  std::copy(x, x + nDim_, xtmp.begin());
  std::vector<double> grad2(nDim_);
  Gradient(x, &grad_[0]);
  for (unsigned int j = 0; j < nDim_; ++j) {
    double step = step_[j] * numDiffStep_;
    if (step == 0)
      step = numDiffStep_;
    xtmp[j] += step;
    Gradient(xtmp.data(), &grad2[0]);
    for (unsigned int i = 0; i < nDim_; ++i) {
      hess_[i * nDim_ + j] = (grad2[i] - grad_[i]) / step;
    }
    xtmp[j] = x[j];
  }
}

extern "C" ROOT::Math::Minimizer *createCeresMinimizer() { return new CeresMinimizer(); }

namespace {
  ROOT::Math::Minimizer *createCeresMinimizer() { return new CeresMinimizer(); }
  struct CeresMinimizerRegister {
    CeresMinimizerRegister() {
      gPluginMgr->AddHandler("ROOT::Math::Minimizer", "Ceres", "", "CeresMinimizer",
                             "createCeresMinimizer()");
    }
  } gCeresMinimizerRegister;
}  // namespace
