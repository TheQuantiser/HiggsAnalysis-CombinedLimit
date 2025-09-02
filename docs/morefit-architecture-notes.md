# MoreFit Architecture Notes

## Combine likelihood flow
- Datacards are parsed into a `RooWorkspace` which contains `RooAbsPdf` models and `RooAbsData` datasets.
- The `combine` CLI drives algorithms derived from `LimitAlgo`.
- `FitterAlgoBase::run` is the common entry point for fits; it creates the NLL via `combineCreateNLL` which wraps `RooAbsPdf::createNLL` and honours the selected evaluation backend.
- The returned `RooAbsReal` is minimised by `CascadeMinimizer`, a convenience wrapper around `RooMinimizer`/Minuit2.  The FCN, gradient and Hessian therefore come from RooFit.
- Typical call chain for a fit diagnostic is
  `datacard → RooWorkspace → FitterAlgoBase::run → combineCreateNLL → RooMinimizer → Minuit2`.

## MoreFit building blocks
- Public headers live in `morefit/include/`:
  - `dimension.hh`/`dimensionvector.hh` describe observable axes.
  - `parametervector.hh`/`parameter` encode fit parameters.
  - `pdf.hh` and `physicspdfs.hh` provide PDF classes (Gaussian, CrystalBall, etc.) and `SumPDF` combiners.
  - `graph.hh` represents symbolic computation graphs that can be JIT‑compiled.
  - `compute.hh`, `compute_opencl.hh`, `compute_llvm.hh` implement evaluation backends; options select OpenCL GPUs or LLVM JIT on CPU.
  - `fitter.hh` contains the Minuit2 adapter and `fitter_options` (minimiser type, analytic gradient/Hessian toggles, optimisation of parameter/event terms).
  - `generator.hh` and `random.hh` expose accelerated toy generation with host or device random numbers.

MoreFit is largely header-only but ships a private copy of **Minuit2** under `morefit/minuit2/`.  For Combine we intend to reuse the Minuit2 library already provided by ROOT and treat the bundled copy as unused.

Key external dependencies are `Eigen3`, `OpenCL`, and `Clang/LLVM` for the JIT backends.

## Backend features
- Evaluation builds a graph once and executes it on the chosen backend.  Parameter‑only and event‑only subgraphs can be cached to minimise recomputation when data or parameters change.
- Analytic gradients and Hessians are available via `fitter_options` (`analytic_gradient`, `analytic_hessian`); numerical derivatives are used if disabled.
- Generator can run on OpenCL devices or threaded LLVM code with configurable seeds and work‑item scheduling.

These notes capture the entry points and main classes needed for integrating MoreFit as an alternative likelihood backend in Combine.
