```shell
cmake . -B build -DVISP_DIR=${PWD}/3rdparty/visp/install/lib/cmake/visp -Daligator_DIR=${PWD}/3rdparty/aligator/install/lib/cmake/aligator/ -Dpinocchio_DIR=${PWD}/3rdparty/pinocchio/install/lib/cmake/pinocchio -Dpinocchio_DIR=${PWD}/3rdparty/proxsuite/install/lib/cmake/proxsuite -DCMAKE_BUILD_TYPE=Release
cmake --build build
export LD_LIBRARY_PATH=$(pwd)/3rdparty/aligator/install/lib:$(pwd)/3rdparty/pinocchio/install/lib:$(pwd)/3rdparty/coal/install/lib:$LD_LIBRARY_PATH
build/viper850-mpc-example -c -d
```
