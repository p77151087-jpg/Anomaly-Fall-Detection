"""Export report figures from verified, frozen real runs without retraining."""

import argparse
import csv
import math
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from fall_detect.config import project_path
from fall_detect.io_utils import ContractError, read_json, sha256, write_json


def verify_frozen(run):
    expected = read_json(run / "frozen-artifacts.json")
    actual = {name: sha256(run / name) for name in expected}
    if actual != expected:
        raise ContractError(f"Frozen artifacts changed: {run.name}")
    return actual


def read_run(run):
    frozen = verify_frozen(run)
    status = read_json(run / "status.json")
    index = read_json(run / "dataset-index.json")
    if status["training"] != "complete" or index["source_kind"] != "real":
        raise ContractError("Report requires a completed real run")
    history_path = run / "training-history.csv"
    with history_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            {
                "epoch": int(r["epoch"]),
                "train_loss": float(r["train_loss"]),
                "val_normal_loss": float(r["val_normal_loss"]),
            }
            for r in csv.DictReader(handle)
        ]
    if [r["epoch"] for r in rows] != list(range(1, status["epochs"] + 1)):
        raise ContractError("Missing or reordered history epochs")
    if not all(
        math.isfinite(r[k]) and r[k] >= 0 for r in rows for k in ("train_loss", "val_normal_loss")
    ):
        raise ContractError("Non-finite or negative loss")
    best = min(rows, key=lambda row: row["val_normal_loss"])
    if best["val_normal_loss"] != status["best_val_normal_loss"]:
        raise ContractError("History and saved best-validation loss disagree")
    counts = {}
    for split in ("train", "validation"):
        entries = [
            index["sequences"][r["sequence_id"]]
            for r in index["split_manifest"]["sequences"]
            if r["split"] == split and r["label"] == 0
        ]
        counts[split] = {
            "planned_sequences": len(entries),
            "usable_sequences": sum(e["valid_features"] > 0 for e in entries),
            "feature_windows": sum(e["valid_features"] for e in entries),
        }
    return {
        "run": run.name,
        "source_kind": "real",
        "epochs": len(rows),
        "best": best,
        "final": rows[-1],
        "counts": counts,
        "history_sha256": sha256(history_path),
        "frozen_artifacts": frozen,
    }, rows


def draw_curve(ax, summary, rows, title):
    epochs = np.array([r["epoch"] for r in rows])
    train = np.array([r["train_loss"] for r in rows])
    validation = np.array([r["val_normal_loss"] for r in rows])
    (line_train,) = ax.plot(epochs, train, color="#1876A3", lw=2.2, label="訓練集（正常活動）")
    (line_val,) = ax.plot(epochs, validation, color="#D97922", lw=2.2, label="驗證集（正常活動）")
    # The displayed points must be the CSV values, with no smoothing or resampling.
    np.testing.assert_array_equal(line_train.get_ydata(), train)
    np.testing.assert_array_equal(line_val.get_ydata(), validation)
    best = summary["best"]
    ax.axvline(best["epoch"], color="#73818D", lw=1, ls=(0, (4, 4)), alpha=0.8)
    ax.scatter(
        best["epoch"],
        best["val_normal_loss"],
        s=55,
        color="#D97922",
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
    )
    ax.annotate(
        f"最佳驗證 Loss\nEpoch {best['epoch']} · {best['val_normal_loss']:.6f}",
        xy=(best["epoch"], best["val_normal_loss"]),
        xycoords="data",
        xytext=(0.52, 0.57),
        textcoords="axes fraction",
        fontsize=10,
        color="#263747",
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": "#73818D", "lw": 0.9},
        bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": "#DCE3E8"},
    )
    ax.set(
        xlabel="訓練回合（Epoch）",
        ylabel="重建誤差（MSE）",
        xlim=(1, summary["epochs"] + 2),
        ylim=(0, max(train.max(), validation.max()) * 1.09),
    )
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=14, color="#203449")
    ax.grid(axis="y", color="#E2E8ED", lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#AFBDC7")
    ax.tick_params(colors="#455866", labelsize=10)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="white",
        framealpha=0.96,
        fontsize=10,
    )


def save_figure(fig, stem):
    files = []
    for suffix in (".png", ".svg"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=300, facecolor="white")
        files.append({"file": path.name, "sha256": sha256(path)})
    plt.close(fig)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="+", default=["runs/openpose18-baseline-001", "runs/le2i-extension-001"]
    )
    parser.add_argument("--output", default="docs/figures/loss")
    args = parser.parse_args()
    runs = [project_path(name) for name in args.runs]
    output = project_path(args.output)
    if any(output.is_relative_to(run) for run in runs):
        raise ContractError("Export outside the frozen run directories")
    font = next(
        (
            name
            for name in ("Microsoft JhengHei", "Noto Sans CJK TC")
            if name in {f.name for f in font_manager.fontManager.ttflist}
        ),
        None,
    )
    if font is None:
        raise ContractError("A Traditional Chinese font is required for these report figures")
    plt.rcParams.update(
        {"font.family": font, "font.size": 11, "axes.unicode_minus": False, "svg.fonttype": "path"}
    )
    output.mkdir(parents=True, exist_ok=True)
    summaries, histories, artifacts = [], [], []
    titles = {
        "openpose18-baseline-001": "原基準 Autoencoder Loss",
        "le2i-extension-001": "Le2i 擴充版 Autoencoder Loss",
    }
    for run in runs:
        summary, rows = read_run(run)
        summaries.append(summary)
        histories.append(rows)
        fig = plt.figure(figsize=(10.6, 6.6))
        ax = fig.add_axes((0.11, 0.28, 0.84, 0.58))
        draw_curve(ax, summary, rows, titles.get(run.name, run.name))
        fig.text(
            0.11,
            0.94,
            f"真實訓練紀錄  /  {run.name}  /  共 {len(rows)} 回合",
            color="#5D6F7C",
            fontsize=10,
        )
        counts = summary["counts"]
        train, val = counts["train"], counts["validation"]
        fig.text(
            0.11,
            0.16,
            f"有效正常資料：Train {train['usable_sequences']} 段／{train['feature_windows']:,} 窗；"
            f"Validation {val['usable_sequences']} 段／{val['feature_windows']:,} 窗。",
            fontsize=10,
            color="#324959",
        )
        note = (
            "Le2i 的 4 段正常驗證影片無有效特徵；此圖的驗證曲線僅來自 UR Fall。"
            if run.name == "le2i-extension-001"
            else "此圖使用 UR Fall 的正常活動資料；不包含 Test Loss。"
        )
        fig.text(0.11, 0.115, note, fontsize=9, color="#5D6F7C")
        fig.text(
            0.11,
            0.07,
            "原始逐回合數值，未平滑。Loss 衡量重建誤差，不是跌倒偵測準確率。",
            fontsize=9,
            color="#5D6F7C",
        )
        artifacts.extend(save_figure(fig, output / f"{run.name}-loss"))
        copied = output / f"{run.name}-history.csv"
        shutil.copyfile(run / "training-history.csv", copied)
        if sha256(copied) != summary["history_sha256"]:
            raise ContractError("Copied history differs from original")
    if len(runs) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
        fig.subplots_adjust(left=0.065, right=0.985, top=0.79, bottom=0.22, wspace=0.25)
        for ax, summary, rows in zip(axes, summaries, histories, strict=True):
            draw_curve(ax, summary, rows, titles.get(summary["run"], summary["run"]))
        fig.text(0.065, 0.94, "兩次真實訓練的收斂紀錄", fontsize=18, color="#203449", weight="bold")
        fig.text(
            0.065,
            0.87,
            "各自顯示 Train／Validation MSE；不直接以跨圖 Loss 大小判斷辨識優劣。",
            fontsize=11,
            color="#5D6F7C",
        )
        fig.text(
            0.065,
            0.105,
            "兩個實驗使用不同的正常訓練資料、Scaler 與特徵設定，縱軸各自縮放。",
            fontsize=10,
            color="#5D6F7C",
        )
        fig.text(
            0.065,
            0.06,
            "擴充版的正常 Validation 仍只有 UR Fall；Le2i 的 4 段正常驗證影片無有效特徵。",
            fontsize=10,
            color="#5D6F7C",
        )
        artifacts.extend(save_figure(fig, output / "loss-side-by-side"))
    for run in runs:
        verify_frozen(run)
    write_json(
        output / "loss-figure-summary.json",
        {
            "source_kind": "real",
            "smoothing": False,
            "test_loss_included": False,
            "train_loss_definition": "sample-weighted online minibatch reconstruction MSE during each epoch",
            "validation_loss_definition": "mean normal validation reconstruction MSE at epoch end",
            "cross_run_loss_directly_comparable": False,
            "frozen_artifacts_unchanged": True,
            "runs": summaries,
            "figures": artifacts,
        },
    )
    print("Exported verified PNG/SVG loss figures and original CSV copies.")


if __name__ == "__main__":
    main()
