Android integration notes — building an APK and on-device personalization

This document summarizes an approach to integrate the exported TFLite model into
an Android app and enable on-device personalization in a resource-friendly way.

1) Use TensorFlow Lite Model Personalization (image tasks)
- TFLite provides guidance and sample code for model personalization: train a small head on-device while keeping the backbone frozen. See the official samples and the "model personalization" docs.

2) Use small models
- MobileNetV2 or EfficientNet-Lite for vision.
- For text, consider extremely small transformers or adapter-only (LoRA/adapters) approaches and keep the trainable parameter count tiny.

3) App UX and safeguards
- Offer these settings in the app: `max_cpu_threads`, `batch_size`, `max_epochs`, `only_while_charging`, `max_wall_seconds`.
- Run a quick resource estimation step before training and refuse or suggest lower settings when memory/CPU/time would exceed safe limits.
- Hook into Android's battery and temperature APIs to pause/resume training.

4) Building an APK for Android 16
- Use Android Studio (or command line Gradle) and target `minSdkVersion 16`.
- Add the TensorFlow Lite AARs and support libraries to the app Gradle dependencies. Prefer `org.tensorflow:tensorflow-lite` and the `support` libraries for model personalization.
- Bundle the exported `.tflite` file in `assets/` and load it with the `Interpreter` or with the personalization APIs.

5) Troubleshooting and realistic expectations
- Full LLM fine-tuning on-device is impractical for mid-range phones. The recommended path is adapter-only personalization or small on-device heads for classification/regression tasks.

Resources
- TensorFlow Lite model personalization guide: https://www.tensorflow.org/lite/examples/model_personalization
- TFLite Android samples: https://github.com/tensorflow/examples/tree/master/lite
