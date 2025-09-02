#include "../interface/CeresMinimizer.h"
#include "../interface/CombineLogger.h"
#include <Math/IMultiGenFunction.h>
#include <Math/MinimizerOptions.h>
#include <TPluginManager.h>
#include <ceres/numeric_diff_cost_function.h>
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
#include <iostream>

CeresMinimizer::CeresMinimizer(const char *) : func_(nullptr), gradFunc_(nullptr), nDim_(0), nFree_(0), nCalls_(0), fMinVal_(0.0), edm_(0.0), numDiffStep_(1e-4), forceNumeric_(false) {}

CeresMinimizer::~CeresMinimizer() {}

void CeresMinimizer::Clear() {
    func_ = nullptr;
    gradFunc_ = nullptr;
    nDim_ = nFree_ = 0;
    nCalls_ = 0;
    x_.clear(); step_.clear(); lower_.clear(); upper_.clear(); isFixed_.clear();
    grad_.clear(); hess_.clear();
    fMinVal_ = 0.0; edm_ = 0.0;
    numDiffStep_ = 1e-4;
    forceNumeric_ = false;
}

void CeresMinimizer::SetFunction(const ROOT::Math::IMultiGenFunction &func) {
    func_ = &func;
    gradFunc_ = dynamic_cast<const ROOT::Math::IMultiGradFunction*>(func_);
    nDim_ = func.NDim();
    nFree_ = nDim_;
    x_.assign(nDim_,0.0);
    step_.assign(nDim_,0.1);
    lower_.assign(nDim_,-std::numeric_limits<double>::infinity());
    upper_.assign(nDim_, std::numeric_limits<double>::infinity());
    isFixed_.assign(nDim_,false);
}

bool CeresMinimizer::SetVariable(unsigned int i, const std::string &, double val, double step) {
    if (i>=nDim_) return false;
    x_[i] = val; step_[i] = step; isFixed_[i]=false; return true;
}

bool CeresMinimizer::SetLimitedVariable(unsigned int i, const std::string &, double val, double step, double lower, double upper) {
    if (i>=nDim_) return false;
    x_[i]=val; step_[i]=step; lower_[i]=lower; upper_[i]=upper; isFixed_[i]=false; return true;
}

bool CeresMinimizer::SetFixedVariable(unsigned int i, const std::string &, double val) {
    if (i>=nDim_) return false;
    x_[i]=val; isFixed_[i]=true; nFree_--; return true;
}

CeresMinimizer::CostFunction::CostFunction(const ROOT::Math::IMultiGradFunction *f) : func(f) {
    set_num_residuals(1);
    mutable_parameter_block_sizes()->push_back(f->NDim());
}

bool CeresMinimizer::CostFunction::Evaluate(double const* const* parameters, double *residuals, double **jacobians) const {
    const double *x = parameters[0];
    double fval = (*func)(x);
    residuals[0] = std::sqrt(fval);
    if (jacobians && jacobians[0]) {
        std::vector<double> grad(func->NDim());
        func->Gradient(x, &grad[0]);
        double coeff = 0.5 / residuals[0];
        for (unsigned int i=0;i<func->NDim();++i) jacobians[0][i] = coeff * grad[i];
    }
    return true;
}

struct NoGradFunctor {
    explicit NoGradFunctor(const ROOT::Math::IMultiGenFunction *f) : func(f) {}
    bool operator()(const double * const x, double *residuals) const {
        residuals[0] = std::sqrt((*func)(x));
        return true;
    }
    const ROOT::Math::IMultiGenFunction *func;
};

bool CeresMinimizer::Minimize() {
    if (!func_) return false;

    // retrieve solver configuration from environment
    int maxIter = 1000;
    if (const char *env = std::getenv("CERES_MAX_ITERATIONS")) maxIter = std::atoi(env);
    std::string linearSolver = std::getenv("CERES_LINEAR_SOLVER") ? std::getenv("CERES_LINEAR_SOLVER") : std::string("dense_qr");
    unsigned int multiStart = 1;
    if (const char *env = std::getenv("CERES_MULTI_START")) multiStart = std::max(1, std::atoi(env));
    double jitter = 1.0;
    if (const char *env = std::getenv("CERES_JITTER")) jitter = std::max(0.0, std::atof(env));
    std::string jitterDist = std::getenv("CERES_JITTER_DIST") ? std::getenv("CERES_JITTER_DIST") : std::string("uniform");
    bool verbose = std::getenv("CERES_VERBOSE") != nullptr;
    bool progress = std::getenv("CERES_PROGRESS") != nullptr;
    int numThreads = 1;
    if (const char *env = std::getenv("CERES_NUM_THREADS")) numThreads = std::max(1, std::atoi(env));
    else if (std::getenv("CERES_AUTO_THREADS")) {
        unsigned hw = std::thread::hardware_concurrency();
        numThreads = hw ? static_cast<int>(hw) : 1;
    }
    unsigned int seed = 12345;
    if (const char *env = std::getenv("CERES_RANDOM_SEED")) seed = static_cast<unsigned int>(std::strtoul(env, nullptr, 10));
    bool forceNumeric = std::getenv("CERES_FORCE_NUMERIC") != nullptr;
    forceNumeric_ = forceNumeric;

    double funTol = ROOT::Math::MinimizerOptions::DefaultTolerance();
    if (const char *env = std::getenv("CERES_FUNCTION_TOLERANCE")) funTol = std::atof(env);
    double gradTol = funTol;
    if (const char *env = std::getenv("CERES_GRADIENT_TOLERANCE")) gradTol = std::atof(env);
    double parTol = funTol;
    if (const char *env = std::getenv("CERES_PARAMETER_TOLERANCE")) parTol = std::atof(env);
    std::string algo = ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo();
    if (const char *env = std::getenv("CERES_ALGO")) algo = env;
    double numericStep = 1e-4;
    if (const char *env = std::getenv("CERES_NUMERIC_DIFF_STEP")) numericStep = std::abs(std::atof(env));
    if (numericStep <= 0) numericStep = 1e-4;
    numDiffStep_ = numericStep;
    std::string diffMethod = std::getenv("CERES_DIFF_METHOD") ? std::getenv("CERES_DIFF_METHOD") : std::string("central");
    std::string lossStr = std::getenv("CERES_LOSS_FUNCTION") ? std::getenv("CERES_LOSS_FUNCTION") : std::string("none");
    double initRadius = 1.0;
    if (const char *env = std::getenv("CERES_INITIAL_RADIUS")) initRadius = std::max(1e-12, std::atof(env));
    double lossScale = 1.0;
    if (const char *env = std::getenv("CERES_LOSS_SCALE")) lossScale = std::max(0.0, std::atof(env));
    std::string logFile = std::getenv("CERES_LOG_FILE") ? std::getenv("CERES_LOG_FILE") : std::string();
    double maxTime = 0.0;
    if (const char *env = std::getenv("CERES_MAX_TIME")) maxTime = std::max(0.0, std::atof(env));
    double boundRelax = 0.0;
    if (const char *env = std::getenv("CERES_BOUND_RELAX")) boundRelax = std::max(0.0, std::atof(env));

    std::vector<double> xbest = x_, xinit = x_;
    double bestFval = std::numeric_limits<double>::infinity();
    ceres::Solver::Summary bestSummary;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0,1.0);
    std::normal_distribution<double> gdist(0.0,1.0);
    bool useGauss = (jitterDist == "gaussian");
    auto startTime = std::chrono::steady_clock::now();

    std::string cfgMsg = Form("config algo=%s linearSolver=%s maxIter=%d funTol=%g gradTol=%g parTol=%g threads=%d",
                              algo.c_str(), linearSolver.c_str(), maxIter, funTol, gradTol, parTol, numThreads);
    CombineLogger::instance().log("CeresMinimizer.cc", __LINE__, cfgMsg, __func__);
    if (verbose) std::cout << "Ceres " << cfgMsg << std::endl;

    unsigned int it = 0;
    for (; it < multiStart; ++it) {
        if (maxTime > 0.0) {
            double elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now()-startTime).count();
            if (elapsed >= maxTime) break;
        }
        if (it > 0) {
            for (unsigned int i = 0; i < nDim_; ++i) {
                double r = useGauss ? gdist(rng) : dist(rng);
                x_[i] = xinit[i] + r * step_[i] * jitter;
            }
        } else {
            x_ = xinit;
        }

        ceres::Problem problem;
        ceres::CostFunction *cost = nullptr;
        if (gradFunc_ && !forceNumeric) cost = new CostFunction(gradFunc_);
        else {
            ceres::NumericDiffOptions ndOpts; ndOpts.relative_step_size = numericStep;
            if (diffMethod == "forward")
                cost = new ceres::NumericDiffCostFunction<NoGradFunctor, ceres::FORWARD, 1, ceres::DYNAMIC>(new NoGradFunctor(func_), ceres::TAKE_OWNERSHIP, nDim_, ndOpts);
            else
                cost = new ceres::NumericDiffCostFunction<NoGradFunctor, ceres::CENTRAL, 1, ceres::DYNAMIC>(new NoGradFunctor(func_), ceres::TAKE_OWNERSHIP, nDim_, ndOpts);
        }
        ceres::LossFunction *loss = nullptr;
        if (lossStr == "huber") loss = new ceres::HuberLoss(lossScale);
        else if (lossStr == "cauchy") loss = new ceres::CauchyLoss(lossScale);
        problem.AddResidualBlock(cost, loss, x_.data());
        for (unsigned int i=0;i<nDim_;++i) {
            if (isFixed_[i]) problem.SetParameterBlockConstant(&x_[i]);
            else {
                if (std::isfinite(lower_[i])) problem.SetParameterLowerBound(x_.data(), i, lower_[i]-boundRelax);
                if (std::isfinite(upper_[i])) problem.SetParameterUpperBound(x_.data(), i, upper_[i]+boundRelax);
            }
        }
        ceres::Solver::Options options;
        options.max_num_iterations = maxIter;
        options.function_tolerance = funTol;
        options.gradient_tolerance = gradTol;
        options.parameter_tolerance = parTol;
        options.minimizer_type = (algo == "LineSearch" ? ceres::LINE_SEARCH : ceres::TRUST_REGION);
        if (linearSolver == "dense_qr") options.linear_solver_type = ceres::DENSE_QR;
        else if (linearSolver == "dense_normal_cholesky") options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        else if (linearSolver == "iterative_schur") options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        else if (linearSolver == "sparse_normal_cholesky") options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        else if (linearSolver == "dense_schur") options.linear_solver_type = ceres::DENSE_SCHUR;
        else if (linearSolver == "sparse_schur") options.linear_solver_type = ceres::SPARSE_SCHUR;
        else {
            CombineLogger::instance().log("CeresMinimizer.cc", __LINE__, Form("Unknown linear solver %s, using dense_qr", linearSolver.c_str()), __func__);
            options.linear_solver_type = ceres::DENSE_QR;
        }
        options.num_threads = numThreads;
        options.initial_trust_region_radius = initRadius;
        options.minimizer_progress_to_stdout = progress;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        double fval = (*func_)(x_.data());
        if (verbose) CombineLogger::instance().log("CeresMinimizer.cc", __LINE__, Form("multi-start %u fval %.6f", it, fval), __func__);
        if (summary.IsSolutionUsable() && fval < bestFval) {
            bestFval = fval;
            xbest = x_;
            bestSummary = summary;
        }
    }
    if (maxTime > 0.0) {
        double elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now()-startTime).count();
        if (elapsed >= maxTime)
            CombineLogger::instance().log("CeresMinimizer.cc", __LINE__, "time limit reached before completing all starts", __func__);
    }

    x_ = xbest;
    nCalls_ = bestSummary.num_successful_steps + bestSummary.num_unsuccessful_steps;
    fMinVal_ = bestFval;
    grad_.assign(nDim_,0.0);
    hess_.assign(nDim_*nDim_,0.0);
    cov_.assign(nDim_*nDim_,0.0);
    err_.assign(nDim_,0.0);
  
    if (gradFunc_) {
        gradFunc_->Gradient(x_.data(), &grad_[0]);
        gradFunc_->Hessian(x_.data(), &hess_[0]);
    } else {
        std::vector<double> xtmp = x_;
        for (unsigned int i=0;i<nDim_;++i) {
            xtmp[i] += numDiffStep_;
            double fp = (*func_)(xtmp.data());
            xtmp[i] -= 2*numDiffStep_;
            double fm = (*func_)(xtmp.data());
            grad_[i] = (fp - fm) / (2*numDiffStep_);
            double f0 = bestFval;
            hess_[i*nDim_+i] = (fp - 2*f0 + fm)/(numDiffStep_*numDiffStep_);
            xtmp[i] = x_[i];
        }
    }

    // compute covariance and parameter errors
    {
        ceres::Problem covProblem;
        ceres::CostFunction *cost = nullptr;
        if (gradFunc_ && !forceNumeric_) cost = new CostFunction(gradFunc_);
        else {
            ceres::NumericDiffOptions ndOpts; ndOpts.relative_step_size = numDiffStep_;
            if (diffMethod == "forward")
                cost = new ceres::NumericDiffCostFunction<NoGradFunctor, ceres::FORWARD, 1, ceres::DYNAMIC>(new NoGradFunctor(func_), ceres::TAKE_OWNERSHIP, nDim_, ndOpts);
            else
                cost = new ceres::NumericDiffCostFunction<NoGradFunctor, ceres::CENTRAL, 1, ceres::DYNAMIC>(new NoGradFunctor(func_), ceres::TAKE_OWNERSHIP, nDim_, ndOpts);
        }
        ceres::LossFunction *loss = nullptr;
        if (lossStr == "huber") loss = new ceres::HuberLoss(lossScale);
        else if (lossStr == "cauchy") loss = new ceres::CauchyLoss(lossScale);
        covProblem.AddResidualBlock(cost, loss, x_.data());
        for (unsigned int i=0;i<nDim_;++i) {
            if (isFixed_[i]) covProblem.SetParameterBlockConstant(&x_[i]);
            else {
                if (std::isfinite(lower_[i])) covProblem.SetParameterLowerBound(x_.data(), i, lower_[i]-boundRelax);
                if (std::isfinite(upper_[i])) covProblem.SetParameterUpperBound(x_.data(), i, upper_[i]+boundRelax);
            }
        }
        ceres::Covariance::Options covOpts;
        ceres::Covariance covariance(covOpts);
        std::vector<std::pair<const double*, const double*> > blocks;
        blocks.emplace_back(x_.data(), x_.data());
        if (covariance.Compute(blocks, &covProblem)) {
            covariance.GetCovarianceBlock(x_.data(), x_.data(), &cov_[0]);
            for (unsigned int i=0;i<nDim_; ++i) err_[i] = std::sqrt(std::fabs(cov_[i*nDim_+i]));
        }
    }


    CombineLogger::instance().log("CeresMinimizer.cc", __LINE__, bestSummary.BriefReport(), __func__);
    if (verbose) CombineLogger::instance().log("CeresMinimizer.cc", __LINE__, bestSummary.FullReport(), __func__);
    if (!logFile.empty()) {
        std::ofstream ofs(logFile.c_str(), std::ios::app);
        if (ofs) ofs << bestSummary.FullReport() << std::endl;
    }

    if (multiStart > 1 && jitter == 0.0)
        CombineLogger::instance().log("CeresMinimizer.cc", __LINE__, "multi-start requested without jitter; results may be identical", __func__);

    return bestSummary.IsSolutionUsable();
}

void CeresMinimizer::Gradient(const double *x, double *grad) const {
    if (gradFunc_ && !forceNumeric_) {
        gradFunc_->Gradient(x, grad);
    } else {
        std::vector<double> xtmp(nDim_);
        std::copy(x, x + nDim_, xtmp.begin());
        for (unsigned int i=0;i<nDim_;++i) {
            xtmp[i] += numDiffStep_;
            double fp = (*func_)(xtmp.data());
            xtmp[i] -= 2*numDiffStep_;
            double fm = (*func_)(xtmp.data());
            grad[i] = (fp - fm)/(2*numDiffStep_);
            xtmp[i] = x[i];
        }
    }
}

void CeresMinimizer::Hessian(const double *x, double *hes) const {
    if (gradFunc_ && !forceNumeric_) {
        gradFunc_->Hessian(x, hes);
    } else {
        std::vector<double> xtmp(nDim_);
        std::copy(x, x + nDim_, xtmp.begin());
        std::vector<double> grad1(nDim_), grad2(nDim_);
        Gradient(x, &grad1[0]);
        for (unsigned int j=0;j<nDim_;++j) {
            xtmp[j] += numDiffStep_;
            Gradient(xtmp.data(), &grad2[0]);
            for (unsigned int i=0;i<nDim_;++i) {
                hes[i*nDim_+j] = (grad2[i]-grad1[i])/numDiffStep_;
            }
            xtmp[j] = x[j];
        }
    }
}

namespace {
    struct CeresMinimizerRegister {
        CeresMinimizerRegister() {
            gPluginMgr->AddHandler("ROOT::Math::Minimizer","Ceres","CeresMinimizer","CeresMinimizer","CeresMinimizer()");
        }
    } gCeresMinimizerRegister;
}
