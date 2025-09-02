# Milestone 0 — Skeleton + Build Toggle

- [ ] **CMake:** add `USE_MOREFIT` option; when ON, include `morefit/` headers and reuse ROOT’s `Minuit2` instead of the bundled copy.
  *Acceptance:* `cmake .. -DUSE_MOREFIT=ON` configures; OFF builds as today; ON builds without modifying any Combine code paths.
- [ ] **Interfaces:** add `interface/ILikelihoodBackend.h` (pure virtual API with `evaluate`, `gradient`, `hessian`, `prepareForDataset`, `prepareForParams`).
- [ ] **RooFit backend:** add `src/RooFitBackend.{h,cc}` implementing `ILikelihoodBackend` using current RooFit NLL evaluation.  This remains the default.
- [ ] **CLI scaffolding:** introduce `src/BackendConfig.{h,cc}` and parse `--backend` in `combine` to store the chosen backend.
- [ ] **Documentation:** update `README.md` describing the `USE_MOREFIT` build toggle and experimental intent.

Completion of these tasks establishes the skeleton for later MoreFit integration without affecting existing behaviour.
