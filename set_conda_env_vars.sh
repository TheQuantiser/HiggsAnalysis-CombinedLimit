conda env config vars set PATH=${PATH}:${PWD}/build/bin
# Include install/lib so that libCeresMinimizer is available at runtime.
conda env config vars set LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/build/lib:${PWD}/install/lib
conda env config vars set PYTHONPATH=${PYTHONPATH}:${PWD}/build/lib/python
