"""Execute all notebook code cells in a fresh Python process; retain outputs.

No cached predictions, models, transformations, or OOFs are loaded. This runner
uses plain Python exec because this notebook contains no IPython magics.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


class Tee:
    def __init__(self, stream, collected):
        self.stream, self.collected = stream, collected

    def write(self, text):
        self.collected.append(text)
        self.stream.write(text)
        self.stream.flush()

    def flush(self):
        self.stream.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", default="gci_home_credit_v18.ipynb")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    path = Path(args.notebook).resolve()
    os.chdir(path.parent)
    archive = path.parent / "runs" / args.name
    archive.mkdir(parents=True, exist_ok=False)
    notebook_bytes = path.read_bytes()
    nb = json.loads(notebook_bytes)
    (archive / "source.ipynb").write_bytes(notebook_bytes)
    namespace = {"__name__": "__main__", "__file__": str(path)}
    count = 0
    started = time.time()
    executed_path = archive / "executed.ipynb"
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        count += 1
        cell["execution_count"] = count
        cell["outputs"] = []
        out, err = [], []
        print(f"CELL {i} START ({time.time()-started:.1f}s)", flush=True)
        try:
            with contextlib.redirect_stdout(Tee(sys.stdout, out)), contextlib.redirect_stderr(Tee(sys.stderr, err)):
                exec(compile("".join(cell["source"]), f"{path.name}:cell_{i}", "exec"), namespace)
        except BaseException as exc:
            cell["outputs"].append(dict(output_type="error", ename=type(exc).__name__,
                                        evalue=str(exc), traceback=traceback.format_exc().splitlines()))
            raise
        finally:
            if out:
                cell["outputs"].insert(0, dict(output_type="stream", name="stdout", text="".join(out)))
            if err:
                cell["outputs"].append(dict(output_type="stream", name="stderr", text="".join(err)))
            executed_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"CELL {i} END ({time.time()-started:.1f}s)", flush=True)
    report = namespace["execution_report"]
    (archive / "execution_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if report["csv_sha256"]:
        csv_path = namespace["output_file"]
        csv_bytes = csv_path.read_bytes()
        assert hashlib.sha256(csv_bytes).hexdigest() == report["csv_sha256"]
        (archive / "submission.csv").write_bytes(csv_bytes)
    print("EXECUTION FINISHED", args.name, flush=True)


if __name__ == "__main__":
    main()
