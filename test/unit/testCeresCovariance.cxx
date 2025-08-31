#include <RooRealVar.h>
#include <RooDataSet.h>
#include <RooGaussian.h>
#include <RooMinimizer.h>
#include <RooFitResult.h>
#include <RooRandom.h>
#include <TMatrixDSym.h>
#include <Math/MinimizerOptions.h>
#include <memory>

int main() {
    RooRealVar x("x","x",-10,10);
    RooRealVar mean("mean","mean",0,-10,10);
    RooRealVar sigma("sigma","sigma",1,0.1,10);
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
    if (cov(0,0) <= 0 || cov(1,1) <= 0) return 1;
    return 0;
}
