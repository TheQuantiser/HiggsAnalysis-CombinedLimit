HiggsAnalysis-CombinedLimit
===========================

### Official documentation

All documentation, including installation instructions, is hosted at
http://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest

The source code of this documentation can be found in the `docs/` folder in this repository.

### Ceres minimizer plugin

The build system integrates the [Ceres Solver](http://ceres-solver.org) as a ROOT minimizer. When the Ceres headers and libraries are available,
`libCeresMinimizer` is built automatically alongside the project. The plugin can be selected at runtime when calling `combine` and supports both
the `TrustRegion` and `LineSearch` algorithms. Use `--cminCeresAlgo` to choose the algorithm explicitly. Available linear solvers include `dense_qr`, `dense_normal_cholesky`, `iterative_schur`, `sparse_normal_cholesky`, `dense_schur` and `sparse_schur`:

```
combine datacard.root --cminDefaultMinimizerType=Ceres --cminCeresAlgo TrustRegion
```

Additional tuning flags are available, for example:

```
combine datacard.root --cminDefaultMinimizerType=Ceres --cminDefaultMinimizerAlgo=TrustRegion \
       --cminCeresMaxIterations 200 --cminCeresLinearSolver dense_normal_cholesky \
       --cminCeresMultiStart 5 --cminCeresJitter 0.5 --cminCeresNumThreads 4 \
       --cminCeresRandomSeed 42 --cminCeresVerbose --cminCeresUseNumericGradient \
       --cminCeresFunctionTolerance 1e-4 --cminCeresGradientTolerance 1e-6 \
       --cminCeresLossFunction huber --cminCeresLossScale 2.0 --cminCeresNumericDiffStep 1e-4 \
       --cminCeresAlgo LineSearch --cminCeresLogFile ceres.log --cminCeresProgress \
       --cminCeresMaxTime 30 --cminCeresJitterDist gaussian --cminCeresBoundRelax 0.01 \
       --cminCeresAutoThreads
```

For convenience the shortcut `--cminUseCeres` selects `--cminDefaultMinimizerType=Ceres --cminDefaultMinimizerAlgo=TrustRegion`.

`--cminCeresLossScale` controls the scale parameter used by robust loss functions and `--cminCeresLogFile` writes the full solver
summary to a file for later inspection. `--cminCeresJitter` tunes the amplitude of the randomised starts used for multi-start fits,
and `--cminCeresUseNumericGradient` forces Ceres to approximate derivatives even when analytic gradients are available. The
`--cminCeresProgress` flag echoes per-iteration updates, `--cminCeresMaxTime` limits the total solve time, `--cminCeresJitterDist`
chooses between uniform and Gaussian perturbations, `--cminCeresBoundRelax` slightly enlarges parameter bounds, and
`--cminCeresAutoThreads` uses all available CPU cores when thread count is not specified.

Multi-start jitter and numeric derivatives can help stabilise fits with flat likelihood surfaces such as those involving many transfer factors.
Linking against Ceres requires the development packages for the solver to be installed and discoverable by CMake.

### Publication 

The `Combine` tool publication can be found [here](https://arxiv.org/abs/2404.06614). Please consider citing this reference if you use the `Combine` tool. 
