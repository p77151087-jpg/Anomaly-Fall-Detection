from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..io_utils import atomic_bytes, write_csv, write_json


def training_curve(path, history, source_kind):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="Train ADL")
    ax.plot(
        [r["epoch"] for r in history],
        [r["val_normal_loss"] for r in history],
        label="Validation ADL",
    )
    ax.set(xlabel="Epoch", ylabel="Reconstruction MSE", title=f"{source_kind}: AE training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def evaluation_report(directory, metrics, rows, windows, events):
    directory = Path(directory)
    write_json(directory / "metrics.json", metrics)
    write_csv(
        directory / "sequence-results.csv",
        rows,
        [
            "sequence_id",
            "label",
            "prediction",
            "sequence_score",
            "unknown_reason",
            "first_alarm_ms",
            "n_scheduled_endpoints",
            "n_valid_endpoints",
            "window_coverage",
        ],
    )
    write_csv(
        directory / "window-results.csv",
        windows,
        [
            "sequence_id",
            "track_id",
            "sample_index",
            "window_id",
            "start_ms",
            "end_ms",
            "available_at_ms",
            "raw_score",
            "event_score",
            "state",
            "reason",
        ],
    )
    write_csv(
        directory / "events.csv",
        events,
        [
            "sequence_id",
            "track_id",
            "confirmed_start_ms",
            "last_active_ms",
            "end_ms",
            "peak_score",
            "end_reason",
        ],
    )
    table = metrics["outcomes_with_unknown"]
    write_csv(
        directory / "outcomes-with-unknown.csv",
        [
            {"truth": label, "normal": row[0], "suspected_fall": row[1], "unknown": row[2]}
            for label, row in zip(["ADL", "Fall"], table)
        ],
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    binary = np.array(table)[:, :2]
    ax.imshow(binary, cmap="Blues")
    for (r, c), value in np.ndenumerate(binary):
        ax.text(
            c,
            r,
            str(value),
            ha="center",
            va="center",
            fontsize=18,
            color="white" if value > binary.max() / 2 else "black",
        )
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Normal", "Suspected fall"],
        yticklabels=["ADL", "Fall"],
        xlabel="Prediction",
        ylabel="Truth",
        title=f"{metrics['source_kind']}: sequence (evaluable n={metrics['n_evaluable']})",
    )
    fig.tight_layout()
    fig.savefig(directory / "confusion-matrix.png", dpi=140)
    plt.close(fig)
    text = (
        f"# 評估結果（{metrics['source_kind']}）\n\n"
        f"單位：sequence。協定：{metrics['protocol_id']}。\n\n"
        f"總數 {metrics['n_total']}，可判定 {metrics['n_evaluable']}，unknown {metrics['n_unknown']}；"
        f"sequence coverage={metrics['sequence_coverage']}，window coverage={metrics['window_coverage']}。\n\n"
        f"Fall F1={metrics['sequence_fall_f1']}，operational accuracy={metrics['operational_accuracy']}。\n\n"
        "synthetic 結果只能證明程式流程，不能作為 UR Fall 成績。AE 只用正常訓練資料；門檻使用含跌倒標籤的 validation。\n\n"
        "本切分是自訂序列隔離，未保證人物隔離。序列 TP 可能來自跌倒前誤報；不能視為事件起始時間正確。"
        "請以 events.csv 確認警示時間。逐幀姿態標籤與事件延遲未列入本次評分。\n\n"
        "## 需要檢視的序列\n\n"
        + "\n".join(
            f"- {r['sequence_id']}: truth={r['label']}, prediction={r['prediction']}, reason={r.get('unknown_reason')}"
            for r in rows
            if r["prediction"] != r["label"]
        )
    )
    atomic_bytes(directory / "evaluation-summary.md", text.encode("utf-8"))
