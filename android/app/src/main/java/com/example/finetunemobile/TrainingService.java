package com.example.finetunemobile;

import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.BatteryManager;
import android.util.Log;

import java.io.IOException;

public class TrainingService {
    public static class TrainingConfig {
        public int maxCpuThreads = 1;
        public int batchSize = 4;
        public int epochs = 1;
        public boolean onlyWhileCharging = true;
    }

    public interface Callback {
        void onProgress(int epoch, float loss);
        void onFinished(boolean success);
    }

    private static volatile boolean abort = false;

    public static void startTraining(Context ctx, TrainingConfig cfg, Callback cb) {
        abort = false;
        new Thread(() -> {
            try {
                if (cfg.onlyWhileCharging && !isCharging(ctx)) {
                    cb.onFinished(false);
                    return;
                }

                TFLiteModelManager mm = new TFLiteModelManager();
                try {
                    // Attempt to load model.tflite from assets. In a real app support loading
                    // from user-provided files or app-private storage.
                    mm.loadModelFromAssets(ctx, "model.tflite", cfg.maxCpuThreads);
                } catch (IOException e) {
                    Log.w("TrainingService", "Model load failed: " + e.getMessage());
                    // Continue with simulated training to demonstrate flow.
                }

                // Launch native finetune binary (LoRA) if available. The native binary should accept:
                // --model <path-to-gguf-or-ggml> --data <path-to-jsonl-or-csv-or-dir> --out <adapter-output>
                // + LoRA hyperparameters. We copy URIs to app-private storage and execute the binary.
                try {
                    // For this implementation we expect MainActivity to have stored selection URIs in files
                    // under ctx.getFilesDir() with known names: "selected_model", "selected_data".
                    java.io.File modelFile = new java.io.File(ctx.getFilesDir(), "selected_model");
                    java.io.File dataFile = new java.io.File(ctx.getFilesDir(), "selected_data");
                    java.io.File outAdapter = new java.io.File(ctx.getFilesDir(), "lora_adapter.bin");

                    String binaryPath = NativeFinetune.prepareBinary(ctx);
                    if (binaryPath == null) {
                        cb.onFinished(false);
                        mm.close();
                        return;
                    }

                    String[] cmd = new String[]{
                            binaryPath,
                            "--model", modelFile.getAbsolutePath(),
                            "--data", dataFile.getAbsolutePath(),
                            "--out", outAdapter.getAbsolutePath(),
                            "--epochs", Integer.toString(cfg.epochs),
                            "--batch_size", Integer.toString(cfg.batchSize),
                            "--threads", Integer.toString(cfg.maxCpuThreads)
                    };

                    NativeFinetune.runProcess(cmd, new NativeFinetune.ProcessCallback() {
                        @Override
                        public void onStdout(String line) {
                            // parse progress lines if the binary prints them
                            // Example: EPOCH 1 LOSS 0.5
                            if (line != null && line.startsWith("EPOCH")) {
                                String[] parts = line.split(" ");
                                try {
                                    int epoch = Integer.parseInt(parts[1]);
                                    float loss = Float.parseFloat(parts[3]);
                                    cb.onProgress(epoch, loss);
                                } catch (Exception ignored) {}
                            }
                        }

                        @Override
                        public void onFinished(int exitCode) {
                            cb.onFinished(exitCode == 0);
                        }
                    });

                } catch (Exception e) {
                    android.util.Log.e("TrainingService", "Native finetune failed", e);
                    cb.onFinished(false);
                }
            } catch (Exception e) {
                Log.e("TrainingService", "Training failed", e);
                cb.onFinished(false);
            }
        }, "ft-training-thread").start();
    }

    public static void abortTraining() {
        abort = true;
    }

    private static boolean isCharging(Context ctx) {
        IntentFilter ifilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
        Intent batteryStatus = ctx.registerReceiver(null, ifilter);
        if (batteryStatus == null) return false;
        int status = batteryStatus.getIntExtra(BatteryManager.EXTRA_STATUS, -1);
        return status == BatteryManager.BATTERY_STATUS_CHARGING || status == BatteryManager.BATTERY_STATUS_FULL;
    }
}
