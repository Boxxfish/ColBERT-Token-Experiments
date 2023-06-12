#!/bin/bash
set -ex

# cd into the colbert_rs directory
cd "$(dirname "$0")"

# build and install the Rust module in the current Python environment
RUSTFLAGS='-C target-cpu=native' maturin develop --release
