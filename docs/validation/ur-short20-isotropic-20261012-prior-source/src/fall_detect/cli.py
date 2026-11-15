import argparse
import json
import sys

from .config import CALIBRATION_OBJECTIVES, load_config, project_path
from .io_utils import ContractError


def parser():
    root = argparse.ArgumentParser(
        prog="python -m fall_detect",
        description="Frozen ST-GCN features + normal-only AE anomaly detection",
    )
    commands = root.add_subparsers(dest="command", required=True)
    loss = commands.add_parser("loss", help="Open the local interactive Loss viewer")
    loss.add_argument("--run", help="Initially selected run directory")
    loss.add_argument("--runs-dir", default="runs", help="Directory containing run folders")
    loss.add_argument("--port", type=int, default=8765)
    loss.add_argument("--no-browser", action="store_true")
    for command, help_text in {
        "doctor": "Check environment, resources and codecs",
        "fetch-assets": "Fetch verified source assets",
        "prepare": "Build split, skeleton and feature caches",
        "train": "Train AE on train ADL only",
        "calibrate": "Select threshold on validation sequences",
        "evaluate": "Evaluate frozen test sequences",
        "demo": "Render a real calibrated run on video/images",
        "smoke-test": "Bounded real or synthetic validation",
    }.items():
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument("--config", help="YAML configuration (paths relative to project root)")
        sub.add_argument("--device", choices=["auto", "cpu", "cuda"])
        if command == "doctor":
            sub.add_argument("--mode", choices=["core", "full"], default="core")
            sub.add_argument("--output")
        elif command == "fetch-assets":
            sub.add_argument("--models", action="store_true")
            sub.add_argument("--dataset", choices=["urfall-cam0-rgb", "le2i-rgb", "caucafall-rgb"])
            sub.add_argument("--limit-per-class", type=int)
        elif command == "prepare":
            sub.add_argument(
                "--reuse-poses-index",
                help="Reuse identical real poses when only resampling hold policy/tolerance changes",
            )
            sub.add_argument(
                "--reuse-index",
                help="Verified reuse from an identical real recipe; creates separate caches",
            )
            sub.add_argument(
                "--sequence-ids",
                nargs="+",
                help="Bounded feature smoke only; incomplete indexes cannot train",
            )
            sub.add_argument(
                "--inventory-only",
                action="store_true",
                help="Validate raw data without model inference",
            )
        elif command == "train":
            sub.add_argument("--run-id", required=True)
            sub.add_argument("--index", help="Prepared dataset index")
            sub.add_argument("--show-loss", action="store_true", help="Show live Loss in browser")
            sub.add_argument("--loss-port", type=int, default=0)
        elif command in ("calibrate", "evaluate", "demo"):
            sub.add_argument("--run", required=True)
            if command == "calibrate":
                sub.add_argument("--objective", choices=CALIBRATION_OBJECTIVES)
                sub.add_argument(
                    "--output-run",
                    help="New run for an explicit alternative calibration; copies trained AE",
                )
            if command == "evaluate":
                sub.add_argument("--split", choices=["test"], default="test")
                sub.add_argument("--output")
            if command == "demo":
                sub.add_argument(
                    "--overlay",
                    choices=["auto", "sample", "actions", "simple", "debug", "presentation"],
                    default="auto",
                    help="auto: independent sample actions + AE history for Human, otherwise simple; sample: require Human; actions: legacy coupled display; simple: AE only; debug: scores; presentation: Chinese panel",
                )
                sub.add_argument("--font", help="Chinese TrueType/OpenType font for presentation")
                sub.add_argument("--input", required=True)
                sub.add_argument("--output", required=True)
                sub.add_argument(
                    "--no-display", action="store_true", help="Headless output (default)"
                )
        elif command == "smoke-test":
            sub.add_argument("--mode", choices=["real", "synthetic"], required=True)
            sub.add_argument("--output")
    return root


def execute(args):
    if args.command == "loss":
        from .visualization.loss_viewer import LossViewer

        run = project_path(args.run) if args.run else None
        if run is not None and not (run / "status.json").is_file():
            raise ContractError(f"Run has no status.json: {run}")
        viewer = LossViewer(
            run.parent if run else project_path(args.runs_dir),
            args.port,
            run.name if run else None,
        )
        viewer.start(open_browser=not args.no_browser).wait()
        return 0
    config = load_config(args.config, args.device)
    if args.command == "doctor":
        from .diagnostics import doctor

        report = doctor(config, args.mode, args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 2
    if args.command == "fetch-assets":
        from .assets import fetch_assets

        if not args.models and not args.dataset:
            raise ContractError("fetch-assets needs --models and/or --dataset urfall-cam0-rgb")
        if args.limit_per_class is not None and args.limit_per_class < 1:
            raise ContractError("--limit-per-class must be positive")
        if args.dataset == "caucafall-rgb":
            from .data.caucafall import fetch_caucafall

            if args.models or args.limit_per_class:
                raise ContractError(
                    "Fetch all CAUCAFall videos separately with its extension config"
                )
            report = fetch_caucafall(config)
        elif args.dataset == "le2i-rgb":
            from .data.le2i import fetch_le2i

            if args.models or args.limit_per_class:
                raise ContractError(
                    "Fetch Le2i separately from models; its pinned split defines the scope"
                )
            report = fetch_le2i(config)
        else:
            report = fetch_assets(config, args.models, bool(args.dataset), args.limit_per_class)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2 if report["errors"] else 0
    if args.command == "prepare":
        from .pipeline import prepare, raw_dataset_report

        if args.inventory_only:
            from .data.splits import dataset_split

            report = raw_dataset_report(config, dataset_split(config), args.sequence_ids)
            print(f"Validated {report['n_validated']}/{report['n_planned']} raw sequences")
            return 0 if report["n_validated"] == report["n_planned"] else 2
        destination, index = prepare(
            config,
            selected_ids=args.sequence_ids,
            reuse_index=project_path(args.reuse_index) if args.reuse_index else None,
            reuse_poses_index=project_path(args.reuse_poses_index)
            if args.reuse_poses_index
            else None,
        )
        print(destination)
        selected = args.sequence_ids or list(index["sequences"])
        return (
            0
            if all(index["sequences"].get(s, {}).get("status") == "prepared" for s in selected)
            else 2
        )
    if args.command == "train":
        import re

        from .training.experiment import train

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
            raise ContractError(
                "run-id may contain only letters, numbers, dots, underscores and hyphens"
            )
        index = (
            project_path(args.index)
            if args.index
            else project_path(config["data"]["cache_dir"]) / "dataset-index.json"
        )
        run = project_path("runs") / args.run_id
        if args.show_loss:
            from .visualization.loss_viewer import LossViewer

            if run.exists():
                raise ContractError(f"Run already exists; choose a new run ID: {run}")
            viewer = LossViewer(run.parent, args.loss_port, run.name).start()
            try:
                print(train(config, index, run))
                print("訓練已完成，Loss 畫面保留；按 Ctrl+C 結束。", flush=True)
                viewer.wait()
            finally:
                viewer.close()
        else:
            print(train(config, index, run))
    elif args.command == "calibrate":
        from .training.experiment import calibrate, derive_calibration_run

        if args.config or args.device:
            raise ContractError(
                "calibrate uses the run's saved configuration; omit --config/--device"
            )
        if bool(args.objective) != bool(args.output_run):
            raise ContractError(
                "Alternative calibration requires both --objective and --output-run"
            )
        run = project_path(args.run)
        if args.output_run:
            run = derive_calibration_run(run, project_path(args.output_run), args.objective)
        print(json.dumps(calibrate(run), ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        from .training.experiment import evaluate

        if args.config or args.device:
            raise ContractError("evaluate uses frozen configuration; omit --config/--device")
        output, _ = evaluate(project_path(args.run), args.split, args.output)
        print(output)
    elif args.command == "demo":
        from .visualization.video import demo

        if args.config:
            raise ContractError("demo uses the run's saved configuration; omit --config")
        print(
            demo(
                project_path(args.run),
                project_path(args.input),
                project_path(args.output),
                args.device,
                overlay_mode=args.overlay,
                font_path=project_path(args.font) if args.font else None,
            )
        )
    elif args.command == "smoke-test":
        from .smoke import real_smoke, synthetic_smoke

        output, report = (real_smoke if args.mode == "real" else synthetic_smoke)(
            config, args.output
        )
        print(json.dumps({"output": str(output), **report}, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 2
    return 0


def main(argv=None):
    # Preserve Chinese paths in console logs without changing the parent shell.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args(argv)
    try:
        return execute(args)
    except (ContractError, FileNotFoundError, ImportError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
