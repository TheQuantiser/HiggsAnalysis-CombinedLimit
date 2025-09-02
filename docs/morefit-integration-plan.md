# MoreFit Integration Plan

## Architecture
- Introduce an abstract `ILikelihoodBackend` with hooks:
  - `prepareForDataset(const RooAbsData*)`
  - `prepareForParams(const RooArgList*)`
  - `double evaluate()`
  - `bool gradient(std::vector<double>&)`
  - `bool hessian(std::vector<double>&)`
- Implementations:
  - **RooFitBackend** – wraps current RooFit objects and preserves existing behaviour (default).
  - **MoreFitBackend** – builds MoreFit computation graphs, selects OpenCL or LLVM backends and exposes analytic derivatives when available.

## Dependency considerations
- Combine already depends on ROOT, which provides RooFit and the **Minuit2** minimiser.
- MoreFit requires Minuit2, Eigen3, Clang/LLVM and OpenCL; it also ships a private Minuit2 copy.
- To avoid duplication, the build will reuse ROOT’s Minuit2 and Eigen3 and ignore the copy under `morefit/minuit2`.

## CMake
- New option `USE_MOREFIT` (OFF by default).
- When ON, locate `LLVM/Clang`, `OpenCL`, and reuse existing `Eigen3`/`ROOT::Minuit2`.
- If dependencies are satisfied, expose `morefit/include` and link against `ROOT::Minuit2`, `clang-cpp`, `LLVM` and `OpenCL`; otherwise emit a warning and disable the backend.
- Add compile definition `USE_MOREFIT` to conditionally build MoreFit code paths.

## Model translation
- A `RooToMoreFitConverter` will map a subset of RooFit classes to MoreFit:
  - Observables `RooRealVar` → `morefit::dimension`.
  - Parameters `RooRealVar` → `morefit::parameter`.
  - Supported PDFs: `Gaussian`, `CrystalBall`, `Exponential`, `Polynomial`, `SumPDF`, and log‑normal constraints.
- Unsupported nodes trigger a warning and fall back to `RooFitBackend`.

## Minuit2 usage
- Combine continues to drive `CascadeMinimizer`/Minuit2.
- `RooFitBackend` passes the usual FCN without derivatives.
- `MoreFitBackend` provides analytic gradient/Hessian to Minuit2 via `fitter_options`; if unavailable the interface returns `false` and the minimiser falls back to numerical derivatives.

## Toy generation
- Expose MoreFit’s `generator` alongside RooFit’s generator.
- Options allow choosing host/device randomisation, OpenCL platform/device, threads and vector width for LLVM.
- Generated datasets are written out in standard ROOT format.

## Runtime selection
- Add command line option `--backend {roofit,morefit}`.
- MoreFit-specific flags: `--mf-backend={opencl,llvm}`, `--mf-opencl-platform`, `--mf-opencl-device`, `--mf-threads`, `--mf-vector-width`, `--mf-analytic-derivatives=on|off`, `--mf-event-precompute=auto|off`.

## Validation & benchmarks
- Parity tests: compare parameter values, errors and FCN between RooFit and MoreFit backends for canonical Combine examples. Require FCN agreement within `1e-6` relative or justify deviations.
- Micro-benchmarks: measure fit time, likelihood scans and toy generation on CPU/GPU; report speed-ups and resource usage.

## Rollout
- Feature flag hidden behind `USE_MOREFIT` and the `--backend` runtime option.
- CI matrix builds with and without MoreFit; failures in MoreFit build are initially non-blocking.
- Documentation and examples accompany each milestone.
