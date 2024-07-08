# Adafel Core

**‼️ Adafel protocol is still under active development with potential breaking changes in future.**

Adafel Protocol is an EVM compatible rollup with onchain Machine Learning capability built using IPC Subnet.

## Prerequisites

On Linux (links and instructions for Ubuntu):

- Install system packages: `sudo apt install build-essential clang cmake pkg-config libssl-dev protobuf-compiler git curl`.
- Install Rust. See [instructions](https://www.rust-lang.org/tools/install).
- Install cargo-make: `cargo install --force cargo-make`.
- Install Docker. See [instructions](https://docs.docker.com/engine/install/ubuntu/).
- Install Foundry. See [instructions](https://book.getfoundry.sh/getting-started/installation).

On MacOS:

- Install Xcode from App Store or terminal: xcode-select --install
- Install Homebrew: https://brew.sh/
- Install dependencies: brew install jq
- Install Rust: https://www.rust-lang.org/tools/install (if you have homebrew installed rust, you may need to uninstall that if you get errors in the build)
- Install Cargo make: cargo install --force cargo-make
- Install docker: https://docs.docker.com/desktop/install/mac-install/
- Install foundry: https://book.getfoundry.sh/getting-started/installation

## Building

```
# make sure that rust has the wasm32 target
rustup target add wasm32-unknown-unknown

# add your user to the docker group
sudo usermod -aG docker $USER && newgrp docker

# clone this repo and build
git clone https://github.com/adafel-official/adafel-core.git
cd ipc
make

# building will generate the following binaries
./target/release/ipc-cli --version
./target/release/fendermint --version
```

## Run tests

```
make test
```

## Code organization

- `ipc/cli`: A Rust binary crate for our client `ipc-cli` application that provides a simple and easy-to-use interface to interact with IPC as a user and run all the processes required for the operation of a subnet.
- `ipc/provider` A Rust crate that implements the `IpcProvider` library. This provider can be used to interact with IPC from Rust applications (and is what the `ipc-cli` uses under the hood).
- `ipc/api`: IPC common types and utils.
- `ipc/wallet`: IPC key management and identity.
- `fendermint`: Peer implementation to run subnets based on Tendermint Core.
- `contracts`: A reference implementation of all the actors (i.e. smart contracts) responsible for the operation of the IPC (Inter-Planetary Consensus) protocol.
- `ipld`: IPLD specific types and libraries

## Machine Learning Precompiles

Adafel protocol achieves its onchain machine learning model training and prediction capability by adding bunch of custom wasm actors and syscalls. Currently the following ML algorithms are implemented:

- Linear Regression
- Logistic Regression
- KNN Regression
- KNN Classification
- Decision Tree Regression
- Decision Tree Classification
- Random Forest Regression
- Random Forest Classification

We have created a solidity library to interact with these syscalls. See [adafel-solidity](https://github.com/adafel-official/adafel-solidity)
