On-device LoRA finetune integration using llama.cpp

Overview
This project uses llama.cpp's finetune tool to perform real on-device LoRA training on small quantized models.

finetune binary CLI contract
The bundled finetune binary (built from llama.cpp) is expected to accept these arguments:

--model <path>          Path to base GGUF model
--data <path>           Path to training data (JSONL format, each line: {"text": "..."})
--out <path>            Output path for LoRA adapter binary
--epochs <n>            Number of training epochs
--batch_size <n>        Batch size per training step
--threads <n>           Number of CPU threads
--lora_rank <r>         LoRA rank (default 8)
--lora_alpha <a>        LoRA alpha (default 16)

Output format
The finetune binary should print progress lines to stdout in the format:
  EPOCH <epoch_num> LOSS <loss_value>

Example:
  EPOCH 1 LOSS 2.345
  EPOCH 2 LOSS 1.987

The app parses these lines to display training progress.

Building the finetune binary
1. Install Android NDK (API 21+) and set ANDROID_NDK_HOME
2. Run: ANDROID_NDK_HOME=/path/to/ndk ./tools/build_native_android.sh
3. Built binaries are placed in: android/app/src/main/assets/binaries/<abi>/finetune
4. Binaries are automatically bundled into the APK.

GitHub Actions CI
The workflow .github/workflows/android-build.yml automatically:
1. Installs the Android NDK
2. Runs build_native_android.sh
3. Packages binaries into APK
4. Uploads APK as artifact

Merging LoRA adapters into .gguf models
After training on-device, a LoRA adapter is saved to app storage. You can merge it into the base model using:

Desktop merge (on-device training output)
1. Basic merge (any finetune tool):
   ```bash
   python tools/merge_lora_to_gguf.py --model base.gguf --adapter adapter.bin --out merged.gguf
   ```
   This tool:
   - Reads GGUF model structure
   - Loads LoRA adapter metadata and tensors
   - Parses adapter parameters (rank, alpha, target layers)
   - Logs merge info to `merged.gguf.merge.json`
   - For full tensor-level merge, integrate with llama.cpp or PyTorch

2. Advanced tensor merge (production):
   ```bash
   python tools/merge_lora_advanced.py --model base.gguf --adapter adapter.bin --out merged.gguf
   ```
   This reference implementation:
   - Extracts LoRA A and B matrices
   - Computes delta = (A @ B) * (alpha / rank)
   - Applies delta to model layers
   - Saves merge statistics
   - Requires checkpoint format knowledge and tensor serialization

3. llama.cpp native merge (if available):
   - llama.cpp has built-in merge functionality
   - See: https://github.com/ggerganov/llama.cpp/blob/master/ggml-lora.md
   - May provide fastest and most robust merging

App merge (on-device)
1. Tap the "Merge" tab in the app
2. Select the trained LoRA adapter (from `app/lora_adapter.bin`)
3. Select base model
4. Optionally specify output path
5. Tap "Merge adapter into model"
6. Merged model is saved to app storage (or specified path)

The MergeService runs merge on a background thread and updates progress in the UI.

Data format
Text training data format (JSONL):
  {"text": "Example sentence one."}
  {"text": "Example sentence two."}
  ...

CSV is also supported if first row is headers. Convert using tools/prepare_dataset.py.

Image dataset format
For image tasks, provide a ZIP or folder with structure:
  class1/image1.jpg
  class1/image2.png
  class2/image3.jpg
  ...

Use tools/prepare_dataset.py to pack and convert.

Security and caveats
- Only run finetune binaries from trusted sources (e.g., official llama.cpp releases).
- On-device training large models (>3B params) is impractical on mid-range phones; use small quantized models.
- Training can be slow and may heat the phone; start with low epochs and batch sizes.

