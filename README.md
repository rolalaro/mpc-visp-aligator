```shell
cmake . -B build -DVISP_DIR=${PWD}/3rdparty/visp/install/lib/cmake/visp -Daligator_DIR=${PWD}/3rdparty/aligator/install/lib/cmake/aligator/ -Dpinocchio_DIR=${PWD}/3rdparty/pinocchio/install/lib/cmake/pinocchio -DCMAKE_BUILD_TYPE=Release
cmake --build build
build/viper850-mpc-example -c -d
```
