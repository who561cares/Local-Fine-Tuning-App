package com.example.finetunemobile;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Build;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Arrays;

public class NativeFinetune {
    public interface ProcessCallback {
        void onStdout(String line);
        void onFinished(int exitCode);
    }

    // Copy a bundled binary from assets to files dir and make it executable.
    public static String prepareBinary(Context ctx) {
        AssetManager am = ctx.getAssets();
        String[] candidates = {
                "binaries/arm64-v8a/finetune",
                "binaries/armeabi-v7a/finetune",
                "binaries/x86_64/finetune"
        };

        for (String cand : candidates) {
            try (InputStream is = am.open(cand)) {
                File dest = new File(ctx.getFilesDir(), "finetune");
                try (OutputStream os = new FileOutputStream(dest)) {
                    byte[] buf = new byte[8192];
                    int r;
                    while ((r = is.read(buf)) >= 0) os.write(buf, 0, r);
                }
                dest.setExecutable(true, true);
                return dest.getAbsolutePath();
            } catch (IOException e) {
                // asset not found, try next
            }
        }
        Log.w("NativeFinetune", "No finetune binary found in assets for supported ABIs");
        return null;
    }

    public static void runProcess(String[] cmd, ProcessCallback cb) throws IOException {
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        Process proc = pb.start();

        // Read stdout in a thread
        new Thread(() -> {
            try (BufferedReader r = new BufferedReader(new InputStreamReader(proc.getInputStream()))) {
                String line;
                while ((line = r.readLine()) != null) {
                    if (cb != null) cb.onStdout(line);
                }
            } catch (IOException ignored) {}
        }, "finetune-stdout-reader").start();

        // wait for completion in another thread and invoke callback
        new Thread(() -> {
            try {
                int code = proc.waitFor();
                if (cb != null) cb.onFinished(code);
            } catch (InterruptedException e) {
                if (cb != null) cb.onFinished(-1);
            }
        }, "finetune-waiter").start();
    }
}
