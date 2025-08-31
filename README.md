HiggsAnalysis-CombinedLimit
===========================

### Official documentation

All documentation, including installation instructions, is hosted at
http://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest

The source code of this documentation can be found in the `docs/` folder in this repository.

### Ceres minimizer plugin

When the [Ceres Solver](http://ceres-solver.org) development package is available, the build system produces a `libCeresMinimizer` plugin that exposes Ceres as a `ROOT::Math::Minimizer`. The plugin can be enabled at runtime by selecting it as the default minimizer. Ceres supports both `TrustRegion` (default) and `LineSearch` algorithms:

```
combine datacard.root --cminDefaultMinimizerType=Ceres --cminDefaultMinimizerAlgo=TrustRegion
```

Linking against Ceres requires the solver to be discoverable by CMake at build time.

### Publication 

The `Combine` tool publication can be found [here](https://arxiv.org/abs/2404.06614). Please consider citing this reference if you use the `Combine` tool. 
