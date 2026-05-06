# GTIRB Toolchain Preparation for SabreL

This document describes how to prepare the GTIRB-based build tools used by SabreL on Ubuntu 24.04.
It covers GTIRB, GTIRB Pretty Printer, and Datalog Disassembly (`ddisasm`).

## Overview

SabreL relies on GTIRB-based rewriting infrastructure for entity-mode binary rewriting and evaluation.
These tools are required for end-to-end binary obfuscation and similarity-model evaluation.

## System Preparation

Install common development packages first:

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config python3 python3-pip \
    protobuf-compiler libprotobuf-dev libboost-all-dev doxygen graphviz \
    libcapstone-dev libglib2.0-dev libssl-dev zlib1g-dev libncurses-dev \
    libsqlite3-dev libffi-dev libgmp-dev libreadline-dev
```

If you need a faster apt mirror, configure a local mirror or use a mirror such as Tsinghua.
The repository does not provide an automatic mirror switcher.

## Build and Install GTIRB

1. Clone the GTIRB repository:

```bash
cd ~
git clone https://github.com/GrammaTech/gtirb.git
git clone https://github.com/GrammaTech/gtirb-pprinter.git
git clone https://github.com/GrammaTech/ddisasm.git
```

2. Install optional Python tooling:

```bash
sudo apt install -y tox
pip3 install tomli
```

3. Build GTIRB:

```bash
cd ~/gtirb
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
cmake --build .
ctest
sudo make install
```

4. Verify the GTIRB installation:

```bash
pkg-config --modversion gtirb
```

## Build and Install GTIRB Pretty Printer

The GTIRB pretty printer is required for converting GTIRB IR back to assembly.

```bash
cd ~/gtirb-pprinter
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_BUILD_TYPE=Release \
      -DGTIRB_PPRINTER_BUILD_TESTING=ON \
      -DCMAKE_CXX_FLAGS="-Wno-error" \
      ..
cmake --build .
ctest
sudo make install
```

If the build fails because CMake cannot find GTIRB, provide an explicit path:

```bash
cmake -DCMAKE_INSTALL_PREFIX=/usr/local -Dgtirb_DIR=/usr/local/lib/cmake/gtirb ..
```

If the shared libraries are not found at runtime, run:

```bash
sudo ldconfig
```

## Build and Install ddisasm

Ddisasm depends on GTIRB, GTIRB pprinter, Capstone, Souffle, libehp, and LIEF.

1. Install the required packages:

```bash
sudo apt install -y flex bison libffi-dev libncurses-dev libsqlite3-dev zlib1g-dev libgmp-dev libreadline-dev
```

2. Install Souffle:

```bash
cd ~
git clone -b 2.4 https://github.com/souffle-lang/souffle
cd souffle
cmake . -Bbuild -DCMAKE_BUILD_TYPE=Release -DSOUFFLE_USE_CURSES=0 -DSOUFFLE_USE_SQLITE=0 -DSOUFFLE_DOMAIN_64BIT=1
cd build
sudo make install -j$(nproc)
```

3. Install libehp:

```bash
cd ~
git clone https://git.zephyr-software.com/opensrc/libehp.git
cd libehp
cmake . -Bbuild
cd build
cmake --build .
sudo cp lib/libehp.so /usr/local/lib/libehp.so
sudo ldconfig
```

4. Install LIEF manually if needed:

```bash
cd ~
wget https://github.com/lief-project/LIEF/releases/download/0.16.6/LIEF-0.16.6-Linux-x86_64.tar.gz
tar -zxvf LIEF-0.16.6-Linux-x86_64.tar.gz
cd LIEF-0.16.6-Linux-x86_64
sudo cp -r include/* /usr/local/include/
sudo cp -r lib/* /usr/local/lib/
sudo cp -r bin/* /usr/local/bin/ 2>/dev/null || true
sudo cp -r share/* /usr/local/share/ 2>/dev/null || true
sudo ldconfig
```

5. Build and install ddisasm:

```bash
cd ~/ddisasm
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_BUILD_TYPE=Release \
      -DGTIRB_PPRINTER_BUILD_TESTING=ON \
      -DCMAKE_CXX_FLAGS="-Wno-error" \
      ..
cmake --build .
ctest
sudo make install
```

6. Verify installation:

```bash
ddisasm --version
```

## Notes

- `sudo ldconfig` is recommended after installing new libraries to refresh the dynamic loader cache.
- If the build toolchain fails on Ubuntu 24.04, confirm that the installed `cmake`, `protobuf`, and `boost` versions meet the minimum requirements.
- This guide is intended for the root SabreL environment, where GTIRB-based rewriting is required for entity-mode obfuscation.
