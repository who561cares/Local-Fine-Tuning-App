package com.example.finetunemobile;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends Activity {
    private static final int MODEL_PICK = 100;
    private static final int DATA_PICK = 101;
    private static final int ADAPTER_PICK = 102;

    private LinearLayout trainSection;
    private LinearLayout mergeSection;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        TextView status = findViewById(R.id.statusText);
        Button pickModel = findViewById(R.id.btnPickModel);
        Button pickData = findViewById(R.id.btnPickData);
        Button startTrain = findViewById(R.id.btnStartTrain);
        EditText epochsInput = findViewById(R.id.inputEpochs);
        EditText batchInput = findViewById(R.id.inputBatch);
        EditText threadsInput = findViewById(R.id.inputThreads);

        Button tabTrain = findViewById(R.id.tabTrain);
        Button tabMerge = findViewById(R.id.tabMerge);
        trainSection = findViewById(R.id.trainSection);
        mergeSection = findViewById(R.id.mergeSection);

        // Tab switching
        tabTrain.setOnClickListener(v -> {
            trainSection.setVisibility(View.VISIBLE);
            mergeSection.setVisibility(View.GONE);
        });

        tabMerge.setOnClickListener(v -> {
            trainSection.setVisibility(View.GONE);
            mergeSection.setVisibility(View.VISIBLE);
        });

        // Train tab
        pickModel.setOnClickListener(v -> openPicker(MODEL_PICK));
        pickData.setOnClickListener(v -> openPicker(DATA_PICK));

        startTrain.setOnClickListener(v -> {
            File modelFile = new File(getFilesDir(), "selected_model");
            File dataFile = new File(getFilesDir(), "selected_data");
            if (!modelFile.exists() || !dataFile.exists()) {
                Toast.makeText(this, "Please pick a model and dataset first.", Toast.LENGTH_LONG).show();
                return;
            }

            int epochs = parseIntOrDefault(epochsInput.getText().toString(), 2);
            int batch = parseIntOrDefault(batchInput.getText().toString(), 4);
            int threads = parseIntOrDefault(threadsInput.getText().toString(), 1);

            status.setText("Starting on-device LoRA finetune...");
            TrainingService.TrainingConfig cfg = new TrainingService.TrainingConfig();
            cfg.maxCpuThreads = threads;
            cfg.batchSize = batch;
            cfg.epochs = epochs;
            cfg.onlyWhileCharging = true;

            TrainingService.startTraining(this, cfg, new TrainingService.Callback() {
                @Override
                public void onProgress(int epoch, float loss) {
                    runOnUiThread(() -> status.setText("Epoch " + epoch + " — loss=" + loss));
                }

                @Override
                public void onFinished(boolean success) {
                    runOnUiThread(() -> {
                        if (success) {
                            status.setText("Training finished! Adapter saved.");
                            status.append("\nAdapter: " + new File(getFilesDir(), "lora_adapter.bin").getAbsolutePath());
                        } else {
                            status.setText("Training failed/aborted");
                        }
                    });
                }
            });
        });

        // Merge tab
        Button pickAdapter = findViewById(R.id.btnPickAdapter);
        Button pickBaseModel = findViewById(R.id.btnPickBaseModel);
        Button startMerge = findViewById(R.id.btnStartMerge);
        EditText outPathInput = findViewById(R.id.inputMergeOut);

        pickAdapter.setOnClickListener(v -> openPicker(ADAPTER_PICK));
        pickBaseModel.setOnClickListener(v -> {
            // Alternatively pick from recent trained adapters. For simplicity, use file picker.
            openPicker(MODEL_PICK);
        });

        startMerge.setOnClickListener(v -> {
            File adapter = new File(getFilesDir(), "selected_adapter");
            File baseModel = new File(getFilesDir(), "selected_model");
            String outPath = outPathInput.getText().toString().trim();
            if (outPath.isEmpty()) outPath = new File(getFilesDir(), "merged_model.gguf").getAbsolutePath();

            if (!adapter.exists()) {
                Toast.makeText(this, "Please pick an adapter.", Toast.LENGTH_LONG).show();
                return;
            }
            if (!baseModel.exists()) {
                Toast.makeText(this, "Please pick base model.", Toast.LENGTH_LONG).show();
                return;
            }

            status.setText("Starting merge...");
            MergeService.MergeConfig cfg = new MergeService.MergeConfig();
            cfg.baseModelPath = baseModel.getAbsolutePath();
            cfg.adapterPath = adapter.getAbsolutePath();
            cfg.outputPath = outPath;

            MergeService.startMerge(this, cfg, new MergeService.MergeCallback() {
                @Override
                public void onProgress(String message) {
                    runOnUiThread(() -> status.setText(message));
                }

                @Override
                public void onFinished(boolean success, String resultPath) {
                    runOnUiThread(() -> {
                        if (success) {
                            status.setText("Merge complete!\nOutput: " + resultPath);
                        } else {
                            status.setText("Merge failed");
                        }
                    });
                }
            });
        });
    }

    private void openPicker(int requestCode) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        startActivityForResult(intent, requestCode);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null) return;
        Uri uri = data.getData();
        if (uri == null) return;

        try {
            File dest = null;
            String msg = "";

            if (requestCode == MODEL_PICK) {
                dest = new File(getFilesDir(), "selected_model");
                msg = "Model saved";
            } else if (requestCode == DATA_PICK) {
                dest = new File(getFilesDir(), "selected_data");
                msg = "Dataset saved";
            } else if (requestCode == ADAPTER_PICK) {
                dest = new File(getFilesDir(), "selected_adapter");
                msg = "Adapter saved";
            }

            if (dest != null) {
                copyUriToFile(uri, dest);
                Toast.makeText(this, msg, Toast.LENGTH_SHORT).show();
            }
        } catch (IOException e) {
            Toast.makeText(this, "Failed to copy file: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void copyUriToFile(Uri uri, File dest) throws IOException {
        try (InputStream in = getContentResolver().openInputStream(uri); OutputStream out = new FileOutputStream(dest)) {
            byte[] buf = new byte[8192];
            int r;
            while ((r = in.read(buf)) >= 0) out.write(buf, 0, r);
        }
    }

    private int parseIntOrDefault(String s, int d) {
        try { return Integer.parseInt(s); } catch (Exception e) { return d; }
    }
}
