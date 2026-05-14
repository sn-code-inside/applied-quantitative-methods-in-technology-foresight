"""
Projects 5 and 6 — TensorFlow/Keras + bean_dataset.zip

This script contains two transfer-learning projects on the same dataset:

Project 5 (Feature Extraction):
- Freeze the pretrained backbone (MobileNetV2) and train only the classification head.

Project 6 (Fine-tuning):
- Start from the best weights from Project 5, unfreeze the last N layers of the backbone,
  and continue training with a smaller learning rate.

------------------------------------------------------------
Dataset setup:
- Place bean_dataset.zip in the same directory as this script or notebook.
- The ZIP contains only class subfolders (no pre-split train/val/test):

    bean_dataset.zip
    ├── angular_leaf_spot/
    ├── bean_rust/
    └── healthy/

- The script automatically splits images into train / validation / test
  using the ratios defined in CFG (default: 70% / 15% / 15%).
- Splitting is deterministic (fixed seed) and stratified per class.

Reproducibility Checklist:
1) Fixed seeds across Python/NumPy/TensorFlow.
2) Best-effort deterministic ops.
3) Deterministic stratified split with fixed seed.
4) Saved configurations + histories + evaluation artifacts to disk.
"""

# ============================
# 0) Imports and reproducibility-first environment flags
# ============================

import os
import json
import random
import zipfile
import shutil
import csv

from dataclasses import dataclass, asdict

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix


# ============================
# 1) Configuration
# ============================

@dataclass
class CFG:
    zip_path: str = "bean_dataset.zip"
    zip_inner_root: str = ""

    raw_dir: str = "./beans_raw"
    data_dir: str = "./beans_data"

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    img_size: int = 160
    batch_size: int = 32
    seed: int = 42

    epochs_a: int = 5
    lr_a: float = 1e-3

    epochs_b: int = 3
    lr_b: float = 1e-5
    unfreeze_last_n: int = 30

    out_dir: str = "./outputs_ch8_projects_5_6"


# ============================
# 2) Reproducibility helpers
# ============================

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def save_run_metadata(cfg: CFG):
    os.makedirs(cfg.out_dir, exist_ok=True)

    meta = {
        "config": asdict(cfg),
        "versions": {
            "python_note": "For exact Python build string, print sys.version.",
            "tensorflow": tf.__version__,
            "keras": tf.keras.__version__ if hasattr(tf.keras, "__version__") else "n/a",
            "numpy": np.__version__,
            "sklearn_note": "If needed, print sklearn.__version__.",
        },
        "determinism_env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
            "TF_CUDNN_DETERMINISTIC": os.environ.get("TF_CUDNN_DETERMINISTIC"),
        },
    }

    with open(
        os.path.join(cfg.out_dir, "run_metadata.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(meta, f, indent=2)


# ============================
# 3) Dataset extraction + stratified split
# ============================

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
}


def get_working_directory():
    """
    Works both in:
    - .py scripts
    - Jupyter notebooks
    - Spyder
    - VSCode interactive sessions
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def extract_zip(cfg: CFG):
    """
    Extract bean_dataset.zip into cfg.raw_dir.
    Skips extraction if images already exist.
    Auto-detects and strips a single top-level wrapper folder.
    """

    script_dir = get_working_directory()

    zip_path = (
        cfg.zip_path
        if os.path.isabs(cfg.zip_path)
        else os.path.join(script_dir, cfg.zip_path)
    )

    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"\nbean_dataset.zip not found at:\n  {zip_path}\n\n"
            f"Please place bean_dataset.zip in:\n  {script_dir}\n"
        )

    already = any(
        os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        for _, _, files in os.walk(cfg.raw_dir)
        for f in files
    )

    if already:
        print(f"Raw dataset already extracted at '{cfg.raw_dir}', skipping.")
        return

    print(f"Extracting {os.path.basename(zip_path)} -> {cfg.raw_dir} ...")

    os.makedirs(cfg.raw_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()

        inner_root = cfg.zip_inner_root

        if not inner_root:
            top_folders = {
                m.split("/")[0]
                for m in members
                if "/" in m
            }

            if len(top_folders) == 1:
                candidate = top_folders.pop()

                if not any(
                    candidate.lower() == cls
                    for cls in (
                        "angular_leaf_spot",
                        "bean_rust",
                        "healthy",
                        "train",
                        "validation",
                        "val",
                        "test",
                    )
                ):
                    inner_root = candidate
                    print(f"  Detected inner root: '{inner_root}' — stripping.")

        for member in members:
            rel = member

            if inner_root and member.startswith(inner_root + "/"):
                rel = member[len(inner_root) + 1:]

            if not rel:
                continue

            target = os.path.join(cfg.raw_dir, rel)

            if member.endswith("/"):
                os.makedirs(target, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target), exist_ok=True)

                with z.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())

    print("Extraction complete.")


def build_split(cfg: CFG):
    """
    Stratified train / validation / test split.
    Skips if the split already exists.
    """

    split_done = all(
        os.path.isdir(os.path.join(cfg.data_dir, split))
        for split in ("train", "validation", "test")
    )

    if split_done:
        print(f"Split already exists at '{cfg.data_dir}', skipping.")
        return

    print("Building stratified train/validation/test split ...")

    rng = random.Random(cfg.seed)

    class_names = sorted([
        d for d in os.listdir(cfg.raw_dir)
        if os.path.isdir(os.path.join(cfg.raw_dir, d))
    ])

    if not class_names:
        raise RuntimeError(
            f"No class folders found in '{cfg.raw_dir}'.\n"
            f"Contents: {os.listdir(cfg.raw_dir)}"
        )

    print(f"  Classes found: {class_names}")

    for cls in class_names:
        cls_src = os.path.join(cfg.raw_dir, cls)

        images = sorted([
            f for f in os.listdir(cls_src)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])

        if not images:
            print(f"  Warning: no images found in class '{cls}', skipping.")
            continue

        rng.shuffle(images)

        n = len(images)
        n_train = int(n * cfg.train_ratio)
        n_val = int(n * cfg.val_ratio)

        splits = {
            "train": images[:n_train],
            "validation": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, files in splits.items():
            dst_dir = os.path.join(cfg.data_dir, split_name, cls)
            os.makedirs(dst_dir, exist_ok=True)

            for fname in files:
                shutil.copy2(
                    os.path.join(cls_src, fname),
                    os.path.join(dst_dir, fname)
                )

        print(
            f"  {cls}: {n} images → "
            f"train={len(splits['train'])}, "
            f"val={len(splits['validation'])}, "
            f"test={len(splits['test'])}"
        )

    print("Split complete.")


def load_split(
    data_dir: str,
    split: str,
    img_size: int,
    batch_size: int,
    augment: bool = False,
    seed: int = 42
):
    split_path = os.path.join(data_dir, split)
    is_train = split == "train"

    ds = tf.keras.utils.image_dataset_from_directory(
        split_path,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=is_train,
        seed=seed,
        label_mode="int",
    )

    class_names = ds.class_names
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    if augment and is_train:
        aug_layer = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.05, seed=seed),
            tf.keras.layers.RandomZoom(0.10, seed=seed),
        ])

        def preprocess_train(x, y):
            x = tf.cast(x, tf.float32)
            x = aug_layer(x, training=True)
            x = preprocess(x)
            return x, y

        ds = ds.map(
            preprocess_train,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    else:
        def preprocess_eval(x, y):
            x = tf.cast(x, tf.float32)
            x = preprocess(x)
            return x, y

        ds = ds.map(
            preprocess_eval,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    return ds.prefetch(tf.data.AUTOTUNE), class_names


def make_datasets(cfg: CFG):
    extract_zip(cfg)
    build_split(cfg)

    ds_train, class_names = load_split(
        cfg.data_dir,
        "train",
        cfg.img_size,
        cfg.batch_size,
        augment=True,
        seed=cfg.seed
    )

    ds_val, _ = load_split(
        cfg.data_dir,
        "validation",
        cfg.img_size,
        cfg.batch_size,
        augment=False,
        seed=cfg.seed
    )

    ds_test, _ = load_split(
        cfg.data_dir,
        "test",
        cfg.img_size,
        cfg.batch_size,
        augment=False,
        seed=cfg.seed
    )

    num_classes = len(class_names)

    print(f"Classes ({num_classes}): {class_names}")

    return ds_train, ds_val, ds_test, num_classes, class_names


# ============================
# 4) Model building
# ============================

def build_model(cfg: CFG, num_classes: int):
    inputs = tf.keras.Input(
        shape=(cfg.img_size, cfg.img_size, 3)
    )

    backbone = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

    x = tf.keras.layers.Dropout(0.2)(backbone.output)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name="beans_mobilenetv2"
    )

    return model, backbone


def compile_model(model: tf.keras.Model, lr: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        ],
    )


def unfreeze_last_layers(backbone: tf.keras.Model, n: int):
    backbone.trainable = True

    for layer in backbone.layers[:-n]:
        layer.trainable = False

    for layer in backbone.layers[-n:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


# ============================
# 5) Evaluation utilities
# ============================

def evaluate_basic(model, ds_test):
    vals = model.evaluate(ds_test, verbose=1)

    return dict(
        zip(model.metrics_names, [float(v) for v in vals])
    )


def get_y_true_pred(model, ds_test):
    y_true_all = []
    y_pred_all = []

    for xb, yb in ds_test:
        probs = model.predict(xb, verbose=0)
        preds = np.argmax(probs, axis=1)

        y_true_all.append(yb.numpy())
        y_pred_all.append(preds.astype(np.int32))

    y_true = np.concatenate(y_true_all).astype(np.int32)
    y_pred = np.concatenate(y_pred_all).astype(np.int32)

    return y_true, y_pred


def save_classification_artifacts(
    out_dir,
    y_true,
    y_pred,
    class_names,
    tag
):
    os.makedirs(out_dir, exist_ok=True)

    report_txt = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True
    )

    txt_path = os.path.join(
        out_dir,
        f"classification_report_{tag}.txt"
    )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    csv_path = os.path.join(
        out_dir,
        f"classification_report_{tag}.csv"
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label",
            "precision",
            "recall",
            "f1-score",
            "support"
        ])

        for label, row in report_dict.items():
            if isinstance(row, dict) and all(
                k in row
                for k in ["precision", "recall", "f1-score", "support"]
            ):
                writer.writerow([
                    label,
                    row["precision"],
                    row["recall"],
                    row["f1-score"],
                    row["support"]
                ])

            elif label == "accuracy":
                writer.writerow([
                    label,
                    "",
                    "",
                    row,
                    ""
                ])

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    npy_path = os.path.join(
        out_dir,
        f"confusion_matrix_{tag}.npy"
    )

    np.save(npy_path, cm)

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm, interpolation="nearest")
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Confusion Matrix (Test Set) — {tag}"
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor"
    )

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()

    png_path = os.path.join(
        out_dir,
        f"confusion_matrix_{tag}.png"
    )

    plt.savefig(png_path, dpi=200)
    plt.close(fig)

    return {
        "classification_report_txt_path": txt_path,
        "classification_report_csv_path": csv_path,
        "confusion_matrix_npy_path": npy_path,
        "confusion_matrix_png_path": png_path,
    }, report_txt


# ============================
# 6) Main experiment runner
# ============================

def main():
    cfg = CFG()

    seed_everything(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)

    save_run_metadata(cfg)

    ds_train, ds_val, ds_test, num_classes, class_names = make_datasets(cfg)

    # ==========================================================
    # Project 5: Feature Extraction
    # ==========================================================

    model_a, backbone_a = build_model(cfg, num_classes)

    backbone_a.trainable = False

    compile_model(model_a, cfg.lr_a)

    out_a = os.path.join(
        cfg.out_dir,
        "project_5_feature_extraction"
    )

    best_a = os.path.join(out_a, "best.keras")

    os.makedirs(out_a, exist_ok=True)

    cb_a = [
        tf.keras.callbacks.ModelCheckpoint(
            best_a,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    print("\n=== PROJECT 5: Feature Extraction (Frozen Backbone) ===")

    hist_a = model_a.fit(
        ds_train,
        validation_data=ds_val,
        epochs=cfg.epochs_a,
        callbacks=cb_a,
        verbose=1
    )

    best_model_a = tf.keras.models.load_model(best_a)

    test_a = evaluate_basic(best_model_a, ds_test)

    y_true_a, y_pred_a = get_y_true_pred(
        best_model_a,
        ds_test
    )

    art_a, rep_txt_a = save_classification_artifacts(
        out_a,
        y_true_a,
        y_pred_a,
        class_names,
        tag="project_5"
    )

    payload_a = {
        "experiment": "Project 5 - Feature Extraction",
        "dataset": {
            "name": "bean_dataset.zip",
            "classes": class_names,
            "split_ratios": {
                "train": cfg.train_ratio,
                "validation": cfg.val_ratio,
                "test": cfg.test_ratio
            }
        },
        "config": asdict(cfg),
        "history": {
            k: [float(x) for x in v]
            for k, v in hist_a.history.items()
        },
        "test_metrics": test_a,
        "artifacts": {
            "best_model_path": best_a,
            **art_a
        }
    }

    with open(
        os.path.join(out_a, "metrics.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(payload_a, f, indent=2)

    print("\n--- FINAL (Project 5) ---")
    print("Test accuracy:", test_a.get("accuracy"))
    print(rep_txt_a)

    # ==========================================================
    # Project 6: Fine-tuning
    # ==========================================================

    model_b, backbone_b = build_model(cfg, num_classes)

    model_b.set_weights(best_model_a.get_weights())

    unfreeze_last_layers(
        backbone_b,
        cfg.unfreeze_last_n
    )

    compile_model(model_b, cfg.lr_b)

    out_b = os.path.join(
        cfg.out_dir,
        "project_6_fine_tuning"
    )

    best_b = os.path.join(out_b, "best.keras")

    os.makedirs(out_b, exist_ok=True)

    cb_b = [
        tf.keras.callbacks.ModelCheckpoint(
            best_b,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    print("\n=== PROJECT 6: Fine-Tuning (Partial Unfreeze + Low LR) ===")

    hist_b = model_b.fit(
        ds_train,
        validation_data=ds_val,
        epochs=cfg.epochs_b,
        callbacks=cb_b,
        verbose=1
    )

    best_model_b = tf.keras.models.load_model(best_b)

    test_b = evaluate_basic(best_model_b, ds_test)

    y_true_b, y_pred_b = get_y_true_pred(
        best_model_b,
        ds_test
    )

    art_b, rep_txt_b = save_classification_artifacts(
        out_b,
        y_true_b,
        y_pred_b,
        class_names,
        tag="project_6"
    )

    payload_b = {
        "experiment": "Project 6 - Fine-Tuning",
        "dataset": {
            "name": "bean_dataset.zip",
            "classes": class_names,
            "split_ratios": {
                "train": cfg.train_ratio,
                "validation": cfg.val_ratio,
                "test": cfg.test_ratio
            }
        },
        "config": asdict(cfg),
        "history": {
            k: [float(x) for x in v]
            for k, v in hist_b.history.items()
        },
        "test_metrics": test_b,
        "artifacts": {
            "best_model_path": best_b,
            **art_b
        }
    }

    with open(
        os.path.join(out_b, "metrics.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(payload_b, f, indent=2)

    print("\n--- FINAL (Project 6) ---")
    print("Test accuracy:", test_b.get("accuracy"))
    print(rep_txt_b)

    # ============================
    # Final message
    # ============================

    print("\nDone.")
    print("Outputs saved to:")
    print(os.path.abspath(cfg.out_dir))

    print("\nProject 5 outputs:")
    print(os.path.abspath(out_a))

    print("\nProject 6 outputs:")
    print(os.path.abspath(out_b))


if __name__ == "__main__":
    main()
