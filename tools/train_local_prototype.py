#!/usr/bin/env python3
"""
Resource-friendly local trainer prototype (image classification)

This script is intended to run on a desktop or a phone development environment to
exercise a small transfer-learning flow and export a TFLite model. It demonstrates
practical safeguards: small batches, limited threads, and epoch limits.

NOTE: This is a prototype. For true on-device training on Android use the
TensorFlow Lite Model Personalization APIs (see android/README.md).
"""
import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def limit_tf_threads(threads: int):
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(threads)


def make_dataset(images_dir: Path, labels_csv: Path, image_size=(128, 128), batch=8):
    import pandas as pd

    df = pd.read_csv(labels_csv)
    filepaths = [str(images_dir / row.filename) for row in df.itertuples(index=False)]
    labels = [row.label for row in df.itertuples(index=False)]
    classes = sorted(list(set(labels)))
    class_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_idx[l] for l in labels], dtype=np.int32)

    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, image_size)
        img = img / 255.0
        return img

    paths_ds = tf.data.Dataset.from_tensor_slices(filepaths)
    img_ds = paths_ds.map(lambda p: load_image(p), num_parallel_calls=1)
    label_ds = tf.data.Dataset.from_tensor_slices(y)
    ds = tf.data.Dataset.zip((img_ds, label_ds))
    ds = ds.shuffle(256).batch(batch).prefetch(1)
    return ds, len(classes)


def build_model(input_shape, num_classes):
    base = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(base.input, out)
    return model


def export_tflite(model: tf.keras.Model, out_path: Path, quantize=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)
    print("Wrote TFLite model:", out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--quantize", action='store_true')
    args = p.parse_args()

    limit_tf_threads(args.threads)

    images_dir = Path(args.images)
    labels_csv = Path(args.labels)
    out_path = Path(args.out)

    ds, num_classes = make_dataset(images_dir, labels_csv, image_size=(128, 128), batch=args.batch)
    model = build_model((128, 128, 3), num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Lightweight training loop with early stopping
    callbacks = [keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]
    model.fit(ds, epochs=args.epochs, callbacks=callbacks)

    export_tflite(model, out_path=out_path, quantize=args.quantize)


if __name__ == '__main__':
    main()
