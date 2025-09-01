#include <RooRealVar.h>
#include <RooDataSet.h>
#include <RooGaussian.h>
#include <RooFormulaVar.h>
#include <RooMinimizer.h>
#include <RooFitResult.h>
#include <RooRandom.h>
#include <TMatrixDSym.h>
#include <Math/MinimizerOptions.h>
#include <memory>
#include <cstdlib>
#include <cmath>

int main() {
    setenv("CERES_COVARIANCE_ALGO", "dense_svd", 1);
    RooRealVar x("x","x",-10,10);
    RooRealVar m1("m1","m1",0,-10,10);
    RooRealVar m2("m2","m2",0,-10,10);
    RooFormulaVar mean("mean","@0+@1",RooArgList(m1,m2));
    RooRealVar sigma("sigma","sigma",1);
    sigma.setConstant(true);
    RooGaussian gauss("gauss","gauss",x,mean,sigma);
    RooDataSet data("data","data",x);
    RooRandom::randomGenerator()->SetSeed(1234);
    for (int i=0; i<100; ++i) { x.setVal(RooRandom::randomGenerator()->Gaus()); data.add(x); }
    std::unique_ptr<RooAbsReal> nll(gauss.createNLL(data));
    RooMinimizer minim(*nll);
    minim.setMinimizerType("Ceres");
    minim.migrad();
    minim.hesse();
    std::unique_ptr<RooFitResult> res(minim.save());
    const TMatrixDSym &cov = res->covarianceMatrix();
    if (cov.GetNrows() != 2) return 1;
    if (!std::isfinite(cov(0,0)) || !std::isfinite(cov(1,1))) return 1;
    return 0;
}
