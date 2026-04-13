set -euo pipefail

ROOT="/home/giovanni-briglia/CLionProjects/deep_forked_okok"
cd "$ROOT"

MODEL="exp/rl_exp/batch0_merged/_models/CC/frontier_policy.onnx"
C1="exp/rl_exp/batch0_merged/_models/CC/test_data/CC_2_2_3__pl_7/RawFiles/hash_merged/000001.dot"
C2="exp/rl_exp/batch0_merged/_models/CC/test_data/CC_2_2_3__pl_7/RawFiles/hash_merged/000002.dot"
C3="exp/rl_exp/batch0_merged/_models/CC/test_data/CC_2_2_3__pl_7/RawFiles/hash_merged/000003.dot"

for f in "$MODEL" "$C1" "$C2" "$C3" "$ROOT/rl_frontier_infer.cpp"; do
  [[ -f "$f" ]] || { echo "Missing file: $f"; exit 1; }
done

# ONNX Runtime dev package location (downloaded locally if missing)
ORT_DIR="${ORT_DIR:-$ROOT/lib/onnxruntime}"
ORT_VERSION="${ORT_VERSION:-1.24.4}"
MASK_LEN="${MASK_LEN:-}"

if [[ ! -f "$ORT_DIR/include/onnxruntime_cxx_api.h" || ! -f "$ORT_DIR/lib/libonnxruntime.so" ]]; then
  echo "ONNX Runtime C++ headers/libs not found, downloading v$ORT_VERSION..."
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT
  URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
  curl -L "$URL" -o "$TMP_DIR/ort.tgz"
  tar -xzf "$TMP_DIR/ort.tgz" -C "$TMP_DIR"
  EXTRACTED="$(find "$TMP_DIR" -maxdepth 1 -type d -name "onnxruntime-linux-x64-*${ORT_VERSION}*" | head -n 1)"
  [[ -n "$EXTRACTED" ]] || { echo "Failed to extract ONNX Runtime"; exit 1; }
  rm -rf "$ORT_DIR"
  mv "$EXTRACTED" "$ORT_DIR"
fi

echo "Compiling rl_frontier_infer..."
g++ -std=c++20 -O2 rl_frontier_infer.cpp \
  -I"$ORT_DIR/include" \
  -L"$ORT_DIR/lib" \
  -lonnxruntime \
  -Wl,-rpath,"$ORT_DIR/lib" \
  -o rl_frontier_infer

echo "Running merged frontier inference..."
EXTRA_ARGS=()
if [[ -n "$MASK_LEN" ]]; then
  EXTRA_ARGS+=(--mask-len "$MASK_LEN")
fi

./rl_frontier_infer \
  --onnx "$MODEL" \
  --mode merged \
  --dataset-type HASHED \
  --candidate "$C1" \
  --candidate "$C2" \
  --candidate "$C3" \
  "${EXTRA_ARGS[@]}"
