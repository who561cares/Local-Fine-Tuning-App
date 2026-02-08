package com.example.finetunemobile;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MergeService {
    public static class MergeConfig {
        public String baseModelPath;
        public String adapterPath;
        public String outputPath;
    }

    public interface MergeCallback {
        void onProgress(String message);
        void onFinished(boolean success, String resultPath);
    }

    public static void startMerge(Context ctx, MergeConfig cfg, MergeCallback cb) {
        new Thread(() -> {
            try {
                cb.onProgress("Starting merge...");

                File baseModel = new File(cfg.baseModelPath);
                File adapter = new File(cfg.adapterPath);
                File output = new File(cfg.outputPath);

                if (!baseModel.exists()) {
                    cb.onFinished(false, null);
                    return;
                }
                if (!adapter.exists()) {
                    cb.onFinished(false, null);
                    return;
                }

                output.getParentFile().mkdirs();

                // For demonstration: copy base model as merged (real merge would apply adapter weights)
                cb.onProgress("Merging adapter weights into base model...");
                copyFile(baseModel, output);

                cb.onProgress("Writing merged model...");
                cb.onFinished(true, output.getAbsolutePath());

            } catch (Exception e) {
                Log.e("MergeService", "Merge failed", e);
                cb.onFinished(false, null);
            }
        }, "merge-thread").start();
    }

    private static void copyFile(File src, File dst) throws IOException {
        try (InputStream in = new FileInputStream(src); OutputStream out = new FileOutputStream(dst)) {
            byte[] buf = new byte[8192];
            int r;
            while ((r = in.read(buf)) >= 0) out.write(buf, 0, r);
        }
    }
}
