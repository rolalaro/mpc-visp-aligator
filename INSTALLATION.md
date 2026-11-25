# Installation Procedure

This document lists the system dependencies and build commands used to set up the `viper850-mpc-example` project.

## System Dependencies

Install the required system packages using `apt`:

```bash
sudo apt-get update
sudo apt-get install -y liburdfdom-headers-dev liburdfdom-dev libboost-dev libboost-filesystem-dev libboost-serialization-dev libassimp-dev
sudo apt-get install -y libfmt-dev libmimalloc-dev
```

## Building 3rd Party Libraries

### 1. COAL
```bash
mkdir -p 3rdparty/coal/build
cmake 3rdparty/coal -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=3rdparty/coal/install -B 3rdparty/coal/build
cmake --build 3rdparty/coal/build -j 4
cmake --install 3rdparty/coal/build
```

### 2. Pinocchio
```bash
mkdir -p 3rdparty/pinocchio/build
cmake 3rdparty/pinocchio -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_COLLISION_SUPPORT=ON -Dcoal_DIR=3rdparty/coal/install/lib/cmake/coal/ -DCMAKE_INSTALL_PREFIX=3rdparty/pinocchio/install -B 3rdparty/pinocchio/build
cmake --build 3rdparty/pinocchio/build -j 2
cmake --install 3rdparty/pinocchio/build
```

### 3. Aligator
```bash
mkdir -p 3rdparty/aligator/build
cmake 3rdparty/aligator -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_PINOCCHO_SUPPORT=ON -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=3rdparty/aligator/install -B 3rdparty/aligator/build
cmake --build 3rdparty/aligator/build
cmake --install 3rdparty/aligator/build
```

### 5. Proxsuite
```bash
mkdir -p 3rdparty/proxsuite/build
cmake 3rdparty/proxsuite -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=3rdparty/proxsuite/install -DBUILD_TESTING=OFF -DBUILD_WITH_VECTORIZATION_SUPPORT=OFF -DBUILD_WITH_OPENMP_SUPPORT=ON  -B 3rdparty/proxsuite/build
cmake --build 3rdparty/proxsuite/build
cmake --install 3rdparty/proxsuite/build
```

## Building the Example Project

```bash
mkdir -p build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=3rdparty/aligator/install:3rdparty/pinocchio/install:3rdparty/coal/install:3rdparty/proxsuite/install build
cmake --build build
```

## Running the Example

To run the executable, ensure `LD_LIBRARY_PATH` includes the installation directories:

```bash
export LD_LIBRARY_PATH=$(pwd)/3rdparty/aligator/install/lib:$(pwd)/3rdparty/pinocchio/install/lib:$(pwd)/3rdparty/coal/install/lib:$LD_LIBRARY_PATH
./build/viper850-mpc-example -c -d
./build/viper850-qp-example -c -d
```
