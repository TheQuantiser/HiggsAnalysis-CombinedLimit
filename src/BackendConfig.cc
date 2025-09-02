#include "BackendConfig.h"

BackendConfig& BackendConfig::instance() {
    static BackendConfig cfg;
    return cfg;
}

void BackendConfig::setBackend(Backend b) { backend_ = b; }

void BackendConfig::setBackendFromString(const std::string& name) {
    if (name == "morefit") backend_ = Backend::MoreFit;
    else backend_ = Backend::RooFit;
}

BackendConfig::Backend BackendConfig::backend() const { return backend_; }

std::string BackendConfig::backendName() const {
    return backend_ == Backend::MoreFit ? "morefit" : "roofit";
}
