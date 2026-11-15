"""Separate fixed two-profile scoring experiment after the completed mirror round."""

import argparse
from pathlib import Path

import normal_data_support as support

from fall_detect.detection.events import score_sequence
from fall_detect.io_utils import ContractError, read_json, sha256, write_csv, write_json
from fall_detect.training import experiment
from fall_detect.training.residual import derive_residual_run

ROOT = support.ROOT
PREFIX = "isotropic-residual-support-20261012"


def evidence(kind):
    return ROOT / "docs/validation" / f"{PREFIX}-{kind}.json"


def freeze():
    if evidence("lock").exists():
        raise ContractError("Preserve fixed scoring protocol")
    support.verify_freeze()
    read_json(support.evidence("validation"))  # Previous round must be finished.
    record = read_json(support.evidence("candidate-lock"))
    choices = [
        {"run": "ur-isotropic-wide-channel-20261012-001", "mode": "train_channel_mse"},
        {"run": "ur-isotropic-wide-stream-20261012-001", "mode": "train_stream_mse"},
    ]
    for choice in choices:
        if (ROOT / "runs" / choice["run"]).exists():
            raise ContractError("New residual run IDs required")
    hashes = {}
    for folder in ["runs/ur-mirror128-20261012-001", "runs/ur-mirror256-20261012-001"]:
        for path in (ROOT / folder).iterdir():
            if path.is_file():
                hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
    support.save_new(
        evidence("lock"),
        {
            **{
                k: record[k]
                for k in (
                    "reference",
                    "reference_metrics",
                    "reference_fn",
                    "coarse_intervals",
                    "selection",
                )
            },
            "source_sha256": sha256(Path(__file__)),
            "choices": choices,
            "preserved_mirror_files": hashes,
            "mirror_result_sha256": sha256(support.evidence("validation")),
            "rationale": "Train-only orientation augmentation lowered normal reconstruction Loss but did not improve Val classification. Test whether existing inverse Train residual channel/stream scaling improves separation on the isotropic wide feature recipe, without refitting AE or using Test. Earlier per_frame residual experiments are a different feature distribution.",
            "stop": "Exactly two fixed score modes, 0 additional training epochs, fixed inverse Train residual rule/floor. Original23 Val only for threshold/selection; no extra weighting, seeds, windows, quality or Test evaluations. Stop after both modes.",
            "graph_limitation": "MCP search/index_status/coverage returned Transport closed after the earlier verified graph session; full bounded residual.py read directly. No exhaustive graph claims.",
            "test_feature_arrays_loaded": 0,
            "new_test_evaluations": 0,
        },
    )
    print("Two residual modes preregistered; zero extra AE epochs", flush=True)


def verify_lock(record):
    support.verify_freeze()
    if sha256(Path(__file__)) != record["source_sha256"]:
        raise ContractError("Scoring source changed after lock")
    if sha256(support.evidence("validation")) != record["mirror_result_sha256"]:
        raise ContractError("Previous round changed")
    for path, expected in record["preserved_mirror_files"].items():
        if sha256(ROOT / path) != expected:
            raise ContractError("Previous mirror run changed")


def derive():
    if evidence("validation").exists():
        raise ContractError("Completed scoring experiment; no extra profiles")
    record = read_json(evidence("lock"))
    verify_lock(record)
    results, accesses = [], []
    with support.prior.train_validation_only(accesses):
        for choice in record["choices"]:
            name = choice["run"]
            run = ROOT / "runs" / name
            derive_residual_run(ROOT / "runs" / record["reference"], run, choice["mode"])
            experiment.calibrate(run)
            result = support.prior.validation_result(name, record, export=False)
            folder = ROOT / "outputs" / PREFIX / name
            folder.mkdir(parents=True)
            write_json(folder / "metrics.json", result["metrics"])
            write_csv(folder / "sequences.csv", result["predictions"])
            windows, events = [], []
            scorer = experiment.Scorer(run)
            for source in scorer.index["split_manifest"]["sequences"]:
                if source["split"] != "validation":
                    continue
                cache = experiment.cache_for(scorer.index, source["sequence_id"])
                _, w, e = score_sequence(
                    cache, scorer.score(cache.features), scorer.threshold["threshold"]
                )
                windows.extend(w)
                events.extend(e)
            write_csv(folder / "windows.csv", windows)
            write_csv(folder / "events.csv", events)
            result["exports"] = {
                p.relative_to(ROOT).as_posix(): sha256(p) for p in folder.iterdir()
            }
            results.append(result)
            print(
                {"run": name, "metrics": result["metrics"], "eligible": result["eligible"]},
                flush=True,
            )
    eligible = [r for r in results if r["eligible"]]
    support.save_new(
        evidence("validation"),
        {
            "results": results,
            "selected_candidate": max(eligible, key=support.prior.rank)["run"]
            if eligible
            else None,
            "additional_ae_epochs": 0,
            "feature_accesses": accesses,
            "test_feature_arrays_loaded": 0,
            "new_test_evaluations": 0,
            "lock_sha256": sha256(evidence("lock")),
            "evidence_tier": "real_feature_smoke",
        },
    )
    verify_lock(record)


def status():
    record = read_json(evidence("lock"))
    verify_lock(record)
    result = read_json(evidence("validation"))
    print(
        {
            "complete": True,
            "selected_candidate": result["selected_candidate"],
            "new_test_evaluations": 0,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "derive", "status"])
    command = parser.parse_args().command
    {"freeze": freeze, "derive": derive, "status": status}[command]()
