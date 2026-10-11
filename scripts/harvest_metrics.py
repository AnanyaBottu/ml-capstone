"""Pull the comparison tables out of the executed notebooks into models/*.csv.

The notebooks already carry their results as stored HTML outputs, so the app
never has to refit anything. Tables are matched on their column signature
rather than a cell index, so inserting cells upstream does not break this.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models"

# name -> (notebook, columns that identify the table, index name[, columns the
# table must not have]). The exclusion exists because the K-Means sweep is a
# strict superset of the Ward sweep's columns.
WANTED = {
    "regression_models":     ("regression",     {"R2", "RMSE", "MAE", "fit (s)"}, "model"),
    "regression_cv":         ("regression",     {"CV mean R2", "CV std"}, "model"),
    "regression_tuning":     ("regression",     {"default R2", "tuned R2"}, "model"),
    "classification_models": ("classification", {"accuracy", "precision_weighted", "roc_auc"}, "model"),
    "classification_timing": ("classification", {"accuracy", "f1_weighted", "fit (s)"}, "model"),
    "classification_cv":     ("classification", {"CV mean F1", "CV std"}, "model"),
    "classification_repr":   ("classification", {"n_features", "f1_weighted"}, "representation"),
    "clustering_models":     ("clustering", {"Silhouette", "Davies-Bouldin", "smallest cluster %"}, "model"),
    "clustering_kmeans_k":   ("clustering", {"inertia", "silhouette", "davies_bouldin"}, "k"),
    "clustering_ward_k":     ("clustering", {"silhouette", "davies_bouldin", "calinski_harabasz"}, "k", {"inertia"}),
    "clustering_agreement":  ("clustering", {"K-Means", "Agglomerative (ward)"}, "model"),
    "clustering_stability":  ("clustering", {"silhouette", "Davies-Bouldin", "Calinski-Harabasz"}, "split"),
    "clustering_shares":     ("clustering", {"train %", "hold-out %"}, "cluster"),
    "clustering_profile":    ("clustering", {"share_%", "cancel_rate", "top segment"}, "cluster"),
    "clustering_external":   ("clustering", {"ARI vs is_canceled", "NMI vs is_canceled"}, "model"),
}


def read_tables(path):
    """Every DataFrame rendered as an execute_result in a notebook."""
    for cell in json.loads(path.read_text()).get("cells", []):
        for out in cell.get("outputs", []):
            html = out.get("data", {}).get("text/html")
            if not html:
                continue
            table = BeautifulSoup("".join(html), "html.parser").find("table")
            if table is None:
                continue
            header = table.select_one("thead tr")
            if header is None:
                continue
            # pandas renders a leading blank cell for the index column
            cols = [c.get_text(strip=True) for c in header.find_all(["th", "td"])][1:]
            rows, index = [], []
            for tr in table.select("tbody tr"):
                cells = tr.find_all(["th", "td"])
                index.append(cells[0].get_text(strip=True))
                rows.append([c.get_text(strip=True) for c in cells[1:]])
            if not rows:
                continue
            yield pd.DataFrame(rows, index=index, columns=cols)


def numeric(col):
    """Convert a column to numbers only if every value parses."""
    parsed = pd.to_numeric(col, errors="coerce")
    return col if parsed.isna().any() else parsed


def main():
    OUT.mkdir(exist_ok=True)
    found = {}
    for nb in {w[0] for w in WANTED.values()}:
        path = ROOT / "notebooks" / f"{nb}.ipynb"
        if not path.exists():
            print(f"skip   {nb}: notebook missing")
            continue
        tables = list(read_tables(path))
        for name, (want_nb, signature, index_name, *rest) in WANTED.items():
            if want_nb != nb:
                continue
            banned = rest[0] if rest else set()
            match = next((t for t in tables if signature <= set(t.columns)
                          and not banned & set(t.columns)), None)
            if match is None:
                print(f"MISS   {name}: no table with columns {sorted(signature)}")
                continue
            match = match.apply(numeric)
            match.index.name = index_name
            match.to_csv(OUT / f"{name}.csv")
            found[name] = match.shape
            print(f"wrote  {name}.csv  {match.shape[0]} rows x {match.shape[1]} cols")

    missing = set(WANTED) - set(found)
    if missing:
        print(f"\n{len(missing)} table(s) not found: {sorted(missing)}", file=sys.stderr)
        print("Re-run the notebook end to end so the outputs are stored.", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
