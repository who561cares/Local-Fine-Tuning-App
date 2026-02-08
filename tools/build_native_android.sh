#!/usr/bin/env bash
set -euo pipefail

echo "=== Building finetune binary for Android ==="

# Store the repository root directory
REPO_ROOT="$(pwd)"
echo "Repository root: $REPO_ROOT"

# Set NDK path using environment variable or fallback
NDK_PATH="${ANDROID_NDK_HOME:-$ANDROID_SDK_ROOT/ndk/$NDK_VERSION}"
export ANDROID_NDK_HOME="$NDK_PATH"
echo "NDK path: $ANDROID_NDK_HOME"

# Create output directory
OUT_DIR="android/app/src/main/assets/binaries/arm64-v8a"
mkdir -p "$OUT_DIR"
echo "Output dir: $REPO_ROOT/$OUT_DIR"

# Clean temp directory
TMP="$(mktemp -d)"
echo "Working in: $TMP"
cd "$TMP"

# Clone llama.cpp with specific commit that supports finetune
echo "Cloning llama.cpp..."
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Checkout specific commit that's known to work with finetune
# Commit b1503 from October 2024 has finetune support
echo "Checking out commit b1503..."
git checkout b1503

# Check for and fix POSIX_MADV issue if file exists
echo "Checking for memory mapping files..."
if find . -name "*.cpp" -type f -exec grep -l "POSIX_MADV" {} \; 2>/dev/null; then
  echo "Found files with POSIX_MADV, applying fix..."
  find . -name "*.cpp" -type f -exec grep -l "POSIX_MADV" {} \; 2>/dev/null | while read -r file; do
    echo "  Fixing $file"
    sed -i 's/POSIX_MADV_WILLNEED/MADV_WILLNEED/g' "$file"
    sed -i 's/POSIX_MADV_RANDOM/MADV_RANDOM/g' "$file"
  done
fi

# Configure build
echo "Configuring CMake..."
mkdir -p build-android
cd build-android

cmake .. \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_METAL=OFF \
  -DLLAMA_CUDA=OFF \
  -DLLAMA_BLAS=OFF \
  -DLLAMA_VULKAN=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_FINETUNE=ON \
  -DBUILD_SHARED_LIBS=OFF

# Try building finetune target
echo "Building finetune..."
if cmake --build . --target finetune -- -j$(nproc); then
  echo "✓ finetune target built successfully"
  FINETUNE_BINARY="bin/finetune"
else
  # If finetune target doesn't exist, build all examples and look for it
  echo "finetune target not found, building all examples..."
  cmake --build . --target examples -- -j$(nproc)
  
  # Look for finetune in various possible locations
  if [ -f "examples/finetune/finetune" ]; then
    FINETUNE_BINARY="examples/finetune/finetune"
  elif [ -f "bin/finetune" ]; then
    FINETUNE_BINARY="bin/finetune"
  elif [ -f "finetune" ]; then
    FINETUNE_BINARY="finetune"
  else
    echo "✗ finetune not found in any expected location!"
    echo "Available executables:"
    find . -type f -executable -name "*" | head -20
    exit 1
  fi
fi

# Check if finetune was built
if [ -f "$FINETUNE_BINARY" ]; then
  echo "✓ finetune built successfully at: $FINETUNE_BINARY"
  echo "File info:"
  file "$FINETUNE_BINARY"
  echo "Size: $(stat -c%s "$FINETUNE_BINARY") bytes"
  
  # Copy to output - use absolute path from repository root
  ABS_OUT_DIR="$REPO_ROOT/$OUT_DIR"
  echo "Copying to: $ABS_OUT_DIR/finetune"
  cp "$FINETUNE_BINARY" "$ABS_OUT_DIR/finetune"
  chmod +x "$ABS_OUT_DIR/finetune"
  
  echo "✓ Binary copied to $ABS_OUT_DIR/finetune"
else
  echo "✗ finetune not found!"
  echo "Checking for other executables..."
  find . -type f -executable -name "*" | head -10
  exit 1
fi

echo "=== Build completed successfully ==="
