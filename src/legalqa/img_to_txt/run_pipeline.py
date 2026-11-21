"""Run this approach from preparation through evaluation; standard library only."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "README.md").is_file() and (p / "scripts").is_dir())
APPROACH = "img_to_txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-index", action="store_true", help="Reuse existing index instead of rebuilding it.")
    parser.add_argument("--rebuild-corpus", action="store_true", help="Recreate structured corpus even if present.")
    parser.add_argument("--batch-size", type=int, default=4, help="Fine-tune training batch size.")
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 trains on the full training split.")
    args = parser.parse_args()
    log_dir = ROOT / "runs" / APPROACH
    log_dir.mkdir(parents=True, exist_ok=True)
    # Append keeps Get-Content -Wait attached across successive runs.
    with (log_dir / "pipeline.log").open("a", encoding="utf-8", buffering=1) as log:
        def report(message):
            line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
            print(line, flush=True)
            log.write(line + "\n")

        def run_stage(number, total, title, command):
            report(f"START stage {number}/{total}: {title}")
            report(subprocess.list2cmdline(command))
            started = time.monotonic()
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True,
                                       encoding="utf-8", errors="replace", bufsize=1)
            messages = queue.Queue()

            def read_output():
                try:
                    for line in process.stdout:
                        messages.put(line.rstrip("\r\n"))
                finally:
                    messages.put(None)

            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            heartbeat = time.monotonic()
            try:
                while True:
                    try:
                        line = messages.get(timeout=1)
                    except queue.Empty:
                        line = ""
                    if line is None:
                        break
                    if line:
                        report(line)
                    if time.monotonic() - heartbeat >= 30:
                        report(f"RUNNING stage {number}/{total}: {title}; elapsed={time.monotonic() - started:.0f}s")
                        heartbeat = time.monotonic()
                code = process.wait()
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                process.wait()
                raise
            finally:
                process.stdout.close()
            if code:
                raise RuntimeError(f"Stage {number}/{total} failed with exit code {code}: {title}")
            report(f"DONE stage {number}/{total}: {title}; elapsed={time.monotonic() - started:.0f}s")

        py = [sys.executable, "-u", "-X", "utf8"]
        stages = []
        corpus = ROOT / "data/structured_corpus/structured_corpus.jsonl"
        if args.rebuild_corpus or not corpus.is_file():
            stages.append(("Build structured corpus", py + ["scripts/build_structured_corpus.py"]))
        rebuild = [] if args.reuse_index else ["--rebuild-index"]
        stages.append(("Build index, dense retrieval, rerank and evaluate public test",
                       py + ["src/legalqa/img_to_txt/evaluate.py", "--device", args.device] + rebuild))
        report(f"BEGIN {APPROACH}; python={sys.executable}")
        try:
            for index, (title, command) in enumerate(stages, 1):
                run_stage(index, len(stages), title, command)
        except KeyboardInterrupt:
            report("INTERRUPTED by user")
            return 130
        except Exception as error:
            report(f"FAILED: {error}")
            return 1
        report("PIPELINE SUCCEEDED: all commands exited successfully; review evaluation results.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

