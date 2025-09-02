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
cmake .. -DUSE_MOREFIT=ON -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
make -j$(nproc)
```

The `CMAKE_PREFIX_PATH` flag allows CMake to locate dependencies within an
active Conda environment. The build also produces a small
`combineMoreFitDemo` executable that prints the result of a dummy evaluation to
confirm the backend was compiled in.

At runtime the backend can be selected with `--backend=morefit`. The feature is
experimental and should not alter existing RooFit behaviour when left at the
default.
