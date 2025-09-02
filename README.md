HiggsAnalysis-CombinedLimit
===========================

### Official documentation

All documentation, including installation instructions, is hosted at
http://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest

The source code of this documentation can be found in the `docs/` folder in this repository.

### Publication 

The `Combine` tool publication can be found [here](https://arxiv.org/abs/2404.06614). Please consider citing this reference if you use the `Combine` tool.

### Experimental MoreFit backend

An optional integration of the [MoreFit](https://github.com/cms-analysis/MoreFit) likelihood engine is provided for
development and benchmarking. It reuses the Minuit2 library already bundled with ROOT to avoid duplication of
dependencies and is disabled by default. To build with the backend enabled:

```
mkdir build && cd build
cmake .. -DUSE_MOREFIT=ON
make -j$(nproc)
```

When a Conda environment is active, CMake automatically consumes
`$CONDA_PREFIX` and adjusts `PATH`, `LD_LIBRARY_PATH`, `LIBRARY_PATH`, and
`CPATH` so that all tools — including ROOT’s `rootcling` — resolve headers and
libraries from the environment.  It also preloads Conda’s `libstdc++` and
`libreadline` to avoid symbol mismatches. You may still pass
`-DCMAKE_PREFIX_PATH` explicitly to override or extend the search path. The
build also produces a small `combineMoreFitDemo` executable that prints the
result of a dummy evaluation to confirm the backend was compiled in.

At runtime the backend can be selected with `--backend=morefit`. The feature is
experimental and should not alter existing RooFit behaviour when left at the
default.
