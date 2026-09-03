"""verify_classify.py -- the gate for the supervised classifier (W4).

Evaluates app/models/classifier.pkl on the test split recorded in the
artifact (missions the model never saw, including unseen severities and
operating points where the corpus grid allows).

Gates, all required:

1. top-1 accuracy >= 0.80 and top-2 >= 0.90. The corpus is simulator data
   with known ground truth; a model that cannot hit these numbers on it is
   not ready.
2. per-class recall >= 0.5 for EVERY class. Averages hide an invisible
   class; this gate makes that impossible.
3. SIGNATURE CONSISTENCY: for each class, at least one of the top-5
   attribution features must be a researched signature feature for that
   fault (fault-signatures.md). A classifier that is accurate but looks at
   the wrong channels has learned a simulator artifact and fails here,
   even with good accuracy. This is the explainability layer doing real
   work, not decoration.

Run after ml_classify.py: python verify_classify.py --corpus ../corpus
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

_APP = Path(__file__).resolve().parent
_ROOT = _APP.parent
sys.path.insert(0, str(_APP))

from ml_classify import load_fault_dataset  # noqa: E402

failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(_ROOT / "corpus"))
    args = ap.parse_args()

    art = joblib.load(_APP / "models" / "classifier.pkl")
    model = art["model"]
    classes = np.array(art["classes"])
    feat_names = art["feature_names"]

    import csv
    with open(Path(args.corpus) / "index.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["label"] == "fault"
                and r["seed"] in art["test_seeds"]]
    check(len(rows) > 0, f"test split: {len(rows)} missions")
    X, y, _ = load_fault_dataset(args.corpus, rows)

    proba = model.predict_proba(X)
    order = np.argsort(-proba, axis=1)
    top1 = classes[order[:, 0]]
    top2 = np.array([[classes[i] for i in row[:2]] for row in order])
    acc1 = float((top1 == y).mean())
    acc2 = float(np.any(top2 == y[:, None], axis=1).mean())
    check(acc1 >= 0.80, f"top-1 accuracy {acc1:.3f} >= 0.80")
    check(acc2 >= 0.90, f"top-2 accuracy {acc2:.3f} >= 0.90")

    print("\nper-class recall (test):")
    worst_cls, worst_rec = None, 1.0
    for cls in classes:
        mask = y == cls
        if mask.sum() == 0:
            continue
        rec = float((top1[mask] == cls).mean())
        if rec < worst_rec:
            worst_cls, worst_rec = cls, rec
        flag = "" if rec >= 0.5 else "   <-- below gate"
        print(f"  {cls:38s} {rec:.3f}  (n={int(mask.sum())}){flag}")
        check(rec >= 0.5, f"class {cls}: recall {rec:.3f} >= 0.50")

    print(f"\nsignature consistency (attribution: "
          f"{art['attribution_method']}):")
    attr = art["attribution"]
    for ci, cls in enumerate(classes):
        top5 = [feat_names[i] for i in np.argsort(-attr[ci])[:5]]
        expected = set(art["expected"].get(cls, []))
        hit = sorted(expected & set(top5))
        ok = len(hit) > 0 if expected else True
        check(ok, f"{cls}: top-5 attribution {top5} "
                  f"{'includes ' + str(hit) if hit else 'MISSING expected ' + str(sorted(expected))}")

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURES -- classifier does not "
              "deploy, rules stay primary")
        sys.exit(1)
    print(f"RESULT: ALL GATES PASSED (worst class: {worst_cls} "
          f"recall {worst_rec:.3f})")


if __name__ == "__main__":
    main()
