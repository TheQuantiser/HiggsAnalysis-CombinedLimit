# MoreFit/Combine Dependency Overview

## Combine
- ROOT (RooFit, RooStats, Minuit2, MathMore)
- Eigen3
- Boost (program_options, filesystem)
- Optional VDT

## MoreFit
- Eigen3
- Clang/LLVM (JIT)
- OpenCL
- Private copy of Minuit2 under `morefit/minuit2/`
- Optional ROOT for benchmarks

## Overlap and plan
- Shared: Minuit2, Eigen3.
- Combine already links ROOT's Minuit2; the MoreFit build will reuse this library and omit the bundled copy to avoid duplication.
- Eigen3 is resolved once and shared.
- Clang/LLVM and OpenCL remain optional extras for MoreFit.
