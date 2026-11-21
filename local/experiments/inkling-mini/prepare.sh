#!/usr/bin/env bash
# Install the pinned Inkling CUDA server for the RTX 4070 SUPER.
set -euo pipefail

here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd -- "$here/../.." && pwd)
runtimes="$root/state/runtimes"
runtime="$runtimes/inkling-mini-6196d98-scales"
revision=6196d9802e0757a5218f16f91d9a6830793ba800
patch="$here/fused-expert-scales.patch"

hash_binaries() (
    cd -- "$1"
    find bin -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum
)

verify() (
    cd -- "$1" || exit 1
    [[ -x bin/llama-server && -s bin/libggml-cuda.so && -s bin/libllama-server-impl.so ]] || exit 1
    cmp -s "$patch" fused-expert-scales.patch || exit 1
    if [[ -f source.revision ]]; then
        [[ $(<source.revision) == "$revision" ]] || exit 1
    else
        # The first measured build had a checksum list and no revision file.
        echo '5e0441cd00cc4534262ef7b22aadd11e1bbfe866662524afe3e8f9bba5bb0583  SHA256SUMS' |
            sha256sum --check --status || exit 1
    fi
    hash_binaries . | cmp -s - SHA256SUMS || exit 1
    for library in bin/*.so.*.*.*; do
        [[ "$library" -ef "${library%%.so.*}.so" && "$library" -ef "${library%.*.*}" ]] || exit 1
    done
)

(( $# <= 1 )) || { echo 'Usage: bash prepare.sh [--verify]' >&2; exit 2; }
case "${1:-}" in
    ''|--verify|--build) ;;
    *) echo 'Usage: bash prepare.sh [--verify]' >&2; exit 2 ;;
esac
if [[ -e "$runtime" || ${1:-} == --verify ]]; then
    verify "$runtime" || { echo "Missing or invalid Inkling runtime at $runtime" >&2; exit 1; }
    echo "Verified $runtime/bin/llama-server"
    exit 0
fi

if [[ ${1:-} != --build ]]; then
    mkdir -p "$runtimes"
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
    export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"
    export HEAVY_PEAKS="$runtimes/inkling-build-peaks"
    capped="$HOME/Code/heavy-management/bin/capped"
    [[ -x "$capped" ]] || capped="$root/libexec/capped"
    exec flock "$runtimes/.inkling-mini-build.lock" "$capped" --mem 16G --time 90m nice -n 10 bash "$here/prepare.sh" --build
fi
scope="/sys/fs/cgroup$(sed -n 's/^0:://p' /proc/self/cgroup)"
[[ $(<"$scope/memory.max") == 17179869184 && $(<"$scope/memory.swap.max") == 0 ]] || {
    echo 'The Inkling build requires the 16 GiB scope with swap disabled.' >&2; exit 1;
}

export PATH="$root/.venv/bin:$PATH"
command -v cmake >/dev/null && command -v ninja >/dev/null || {
    echo 'Install cmake and ninja with local/bin/setup-localmodel --run.' >&2; exit 1;
}
shopt -s nullglob
toolkit="${CUDAToolkit_ROOT:-${CUDA_HOME:-}}"
candidates=("$toolkit")
if [[ -z "$toolkit" ]]; then
    nvcc=$(command -v nvcc || true)
    candidates=("$root"/dory/.venv/lib/python*/site-packages/nvidia/cu13
        "$root/state/builds/strata-cache-route-20261109/cuda-site/nvidia/cu13" /usr/local/cuda "${nvcc%/bin/nvcc}")
fi
cudart=() cublas=() cublas_lt=()
for candidate in "${candidates[@]}"; do
    [[ -n "$candidate" && -x "$candidate/bin/nvcc" ]] || continue
    candidate=$(realpath -- "$candidate")
    for lib in "$candidate/lib" "$candidate/lib64" "$candidate/lib/x86_64-linux-gnu"; do
        cudart=("$lib"/libcudart.so*) cublas=("$lib"/libcublas.so*) cublas_lt=("$lib"/libcublasLt.so*)
        if (( ${#cudart[@]} && ${#cublas[@]} && ${#cublas_lt[@]} )); then
            toolkit=$candidate
            break 2
        fi
    done
done
(( ${#cudart[@]} && ${#cublas[@]} && ${#cublas_lt[@]} )) || {
    echo 'Set CUDAToolkit_ROOT to a toolkit containing nvcc, libcudart and cuBLAS.' >&2; exit 1;
}

stage=$(mktemp -d -- "$runtimes/.inkling-mini-XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
source_dir="$stage/source"
git init --quiet "$source_dir"
git -C "$source_dir" fetch --depth=1 https://github.com/ggml-org/llama.cpp.git "$revision"
git -C "$source_dir" checkout --quiet --detach FETCH_HEAD
[[ $(git -C "$source_dir" rev-parse HEAD) == "$revision" ]]
git -C "$source_dir" apply "$patch"

driver=()
[[ ! -f /usr/lib/wsl/lib/libcuda.so.1 ]] || driver=(-DCUDA_cuda_driver_LIBRARY=/usr/lib/wsl/lib/libcuda.so.1)
cmake -S "$source_dir" -B "$stage/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MAKE_PROGRAM="$(command -v ninja)" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89-real \
    -DCMAKE_CUDA_COMPILER="$toolkit/bin/nvcc" -DCUDAToolkit_ROOT="$toolkit" \
    -DCUDA_cudart_LIBRARY="${cudart[0]}" -DCUDA_cublas_LIBRARY="${cublas[0]}" \
    -DCUDA_cublasLt_LIBRARY="${cublas_lt[0]}" "${driver[@]}" -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH="\$ORIGIN;$lib" -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF -DGGML_CUDA_NCCL=OFF -DGGML_CCACHE=OFF
cmake --build "$stage/build" --target llama-server --parallel 5

install="$stage/install"
mkdir -p "$install/bin"
cp -a "$stage/build/bin/llama-server" "$stage/build/bin/"*.so* "$install/bin/"
cp "$patch" "$install/"
echo "$revision" > "$install/source.revision"
hash_binaries "$install" > "$install/SHA256SUMS"
verify "$install"
mv -T -- "$install" "$runtime"
echo "Installed $runtime/bin/llama-server"
