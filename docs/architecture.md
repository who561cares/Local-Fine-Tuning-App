Architecture and design notes — resource friendly on-device fine-tuning

Goals
- Support both text and image personalization for medium-range phones (4–8 cores, ~6–11GB RAM).
- Reduce training time and memory by using small models, transfer learning, and parameter-efficient fine-tuning.
- Prevent overheating and resource exhaustion by limiting CPU threads, using small batches, throttling work, and allowing the user to schedule/preview training.

Key recommendations
- Image tasks: use TensorFlow Lite + Model Personalization. Choose lightweight backbones (MobileNetV2, EfficientNet-Lite). Train only a small classification/regression head on-device; keep most parameters frozen.
- Text tasks: use extremely small transformer or RNN models converted to TFLite, or perform adapter-style fine-tuning where only a small set of parameters (adapters/LoRA) are trained. Full transformer training on-device is typically infeasible on mid-range phones.
- Quantization: use 8-bit or 16-bit quantized models where possible to reduce memory and speed up inference and training.
- Checkpointing: save and resume training frequently; keep checkpoint sizes small (only adapter/head weights where applicable).

Safeguards
- CPU throttling: set TensorFlow threading (intra/inter ops) and allow a max-CPU config option.
- Thermal/UX: expose UI to pause/resume and a scheduler (run only while charging / when temperature OK).
- Memory: limit batch size and use streaming dataset iterators; abort with clear error and guidance when memory is insufficient.

Tradeoffs and constraints
- True full-model fine-tuning of large LLMs on-device is not practical; use PEFT (adapters/LoRA) or server-assisted fine-tuning for larger models.
- TFLite supports on-device training for a subset of op types and workflows; image personalization is well-supported, text is progressing but limited.

Recommended flow
1. User uploads dataset (csv/json/jsonl for text/tabular, image files for vision).
2. Local converter packages data into a compact TF-friendly format.
3. User configures training args (epochs, batch size, CPU threads, max wall time, run-only-while-charging).
4. App runs a short validation pass to estimate time/memory and refuses or suggests lower settings if unsafe.
5. Training runs with periodical checkpointing and temperature/CPU monitoring hooks.
