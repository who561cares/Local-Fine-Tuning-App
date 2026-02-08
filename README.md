Resource-friendly on-device fine-tuning with LoRA merging

Overview
- Real on-device LoRA training optimized for medium-range Android phones (4–8 cores, ~6–11GB RAM).
- Supports text and image model training using small quantized GGUF models (recommended: ≤3B params).
- Includes desktop and on-device tools to train LoRA adapters and merge them back into base models.
- Uses llama.cpp finetune binary (cross-compiled for Android) for efficient on-device training.

Contents
- `docs/architecture.md` — design constraints and recommended model sizes
- `docs/native_integration.md` — finetune binary CLI contract, build, and merge instructions
- `tools/prepare_dataset.py` — convert csv/json/jsonl and image folders to training formats
- `tools/train_local_prototype.py` — desktop LoRA training prototype (Keras + TFLite export)
- `tools/merge_lora_to_gguf.py` — merge trained LoRA adapters into base GGUF models
- `tools/build_native_android.sh` — cross-compile llama.cpp finetune for Android ABIs
- `android/` — app source with on-device training and merge UI

Quick start

Desktop prototype and merge
1. Set up environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Prepare data:
   ```bash
   python tools/prepare_dataset.py --input data/mydata.csv --out data/processed --type tabular
   ```

3. Train locally:
   ```bash
   python tools/train_local_prototype.py --images data/processed/images --labels data/processed/labels.csv --out model_desktop.tflite --epochs 3
   ```

4. Merge LoRA adapter (after on-device training):
   ```bash
   python tools/merge_lora_to_gguf.py --model base.gguf --adapter lora_trained.bin --out merged.gguf
   ```

Android app (on-device training)
1. Build the Android NDK finetune binary:
   ```bash
   ANDROID_NDK_HOME=/path/to/ndk ./tools/build_native_android.sh
   ```

2. Build the APK:
   ```bash
   cd android
   ./gradlew assembleDebug
   ```

3. Or use GitHub Actions CI:
   - Push to GitHub and run the workflow to auto-build NDK binaries and APK.
   - Download the APK artifact from the workflow run.

4. Install and use:
   - Open app, select a GGUF model and training data (JSONL/CSV/ZIP).
   - Configure epochs, batch size, and max CPU threads.
   - Tap "Start on-device LoRA finetune".
   - After training, use the "Merge" tab to blend the adapter into the base model.

Supported formats
- Text data: CSV, JSON, JSONL (each record should contain text)
- Image data: JPG, PNG (organized in class subfolders or ZIPped)
- Models: GGUF (recommended), TensorFlow Lite, or PyTorch-compatible formats

Recommended models
- For text: Mistral-7B-Q4 or similar quantized LLMs in GGUF format
- For images: MobileNetV2, EfficientNet-Lite (TFLite or GGUF)
- All models should be quantized (4-bit or 8-bit) for resource efficiency

Building the APK
See `android/README.md` and `docs/native_integration.md` for detailed APK build and native binary setup.

Next steps
- Review docs/native_integration.md for finetune binary CLI details
- Adapt merge_lora_to_gguf.py to your chosen finetune tool if needed
- Test on-device training with small models first (1–3B params)
# Local-Fine-Tuning-App
