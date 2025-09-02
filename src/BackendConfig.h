#ifndef HIGGSANALYSIS_COMBINEDLIMIT_BACKENDCONFIG_H
#define HIGGSANALYSIS_COMBINEDLIMIT_BACKENDCONFIG_H

#include <string>

class BackendConfig {
public:
    enum class Backend { RooFit, MoreFit };

    static BackendConfig& instance();

    void setBackend(Backend b);
    void setBackendFromString(const std::string& name);
    Backend backend() const;
    std::string backendName() const;

private:
    Backend backend_{Backend::RooFit};
};

#endif
