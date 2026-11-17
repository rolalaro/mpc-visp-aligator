```shell
cd build
cmake .. -DVISP_DIR=~/visp_ws/build-mpc-ws -Daligator_DIR=~/visp_ws/3rdparty/aligator/install/lib/cmake/aligator/ -Dpinocchio_DIR=~/visp_ws/3rdparty/pinocchio/install/lib/cmake/pinocchio -DCMAKE_BUILD_TYPE=Release
make -j10
```
