#!/usr/bin/env python3
# read_filtered_primes.py
#
# Scan graphs dir for misr_filtered_*.json files and extract H_prime / V_prime.
# Outputs:
#   - hprime_vprime_bundle.json (structured list)
#   - hprime_vprime_bundle.csv  (flat/compact)
#
# Usage:
#   python read_filtered_primes.py --graphs_dir graphs/n12
#   (all args are optional; see --help)

import os
import re
import json
import argparse
from glob import glob
from typing import Any, Dict, List, Tuple

def _coerce_int_list(x: Any) -> List[int]:
    if isinstance(x, list):
        out = []
        for v in x:
            if isinstance(v, bool):
                # avoid treating bools as ints if corrupted
                raise ValueError("Boolean found in H_prime/V_prime")
            if isinstance(v, (int, float)):
                out.append(int(v))
            else:
                # allow stringified ints
                out.append(int(str(v)))
        return out
    raise ValueError("Expected list for H_prime/V_prime")

def load_filtered_file(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    # expected keys in our writer: "instance_index", "instance_key", "H_prime", "V_prime"
    idx = data.get("instance_index")
    key = data.get("instance_key", "")
    Hp = data.get("H_prime", [])
    Vp = data.get("V_prime", [])
    try:
        Hp = _coerce_int_list(Hp)
        Vp = _coerce_int_list(Vp)
    except Exception as e:
        raise ValueError(f"{os.path.basename(path)} has invalid H_prime/V_prime: {e}")
    return {
        "idx": idx,
        "instance_key": key,
        "H_prime": Hp,
        "V_prime": Vp,
        "source": path
    }

def numeric_idx_from_filename(path: str) -> int:
    # matches ..._NNN.json, returns NNN as int if present; else large fallback for stable sort
    m = re.search(r'_(\d{1,5})\.json$', os.path.basename(path))
    return int(m.group(1)) if m else 10**9

def main():
    ap = argparse.ArgumentParser(description="Extract H_prime and V_prime from misr_filtered_*.json bundle.")
    ap.add_argument("--graphs_dir", default="graphs/n12", help="Directory that contains misr_filtered_*.json")
    ap.add_argument("--pattern", default="misr_filtered_*.json", help="Glob pattern for filtered jsons")
    ap.add_argument("--out_json", default="hprime_vprime_bundle.json", help="Output JSON filename (written to graphs_dir)")
    ap.add_argument("--out_csv",  default="hprime_vprime_bundle.csv",  help="Output CSV filename (written to graphs_dir)")
    ap.add_argument("--require_nonempty", action="store_true",
                    help="If set, skip entries with empty H_prime or V_prime")
    args = ap.parse_args()

    in_dir = args.graphs_dir
    files = sorted(glob(os.path.join(in_dir, args.pattern)), key=numeric_idx_from_filename)
    if not files:
        print(f"[WARN] No files matched {os.path.join(in_dir, args.pattern)}")
        return

    rows: List[Dict[str, Any]] = []
    for p in files:
        try:
            rec = load_filtered_file(p)
            if args.require_nonempty and (len(rec["H_prime"]) == 0 or len(rec["V_prime"]) == 0):
                continue
            rows.append(rec)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")

    if not rows:
        print("[INFO] Nothing to write after filtering.")
        return

    # Write JSON bundle
    out_json_path = os.path.join(in_dir, args.out_json)
    with open(out_json_path, "w") as f:
        json.dump({"items": rows}, f, indent=2)
    print(f"[OK] JSON written: {out_json_path} ({len(rows)} items)")

    # Write CSV
    out_csv_path = os.path.join(in_dir, args.out_csv)
    with open(out_csv_path, "w") as f:
        # H'/V' are lists; store as semicolon-joined to keep CSV compact
        f.write("idx,instance_key,H_prime,V_prime,source\n")
        for r in rows:
            hp = ";".join(map(str, r["H_prime"]))
            vp = ";".join(map(str, r["V_prime"]))
            idx = "" if r["idx"] is None else str(r["idx"])
            key = r["instance_key"] or ""
            src = r["source"]
            # escape commas inside fields if any (unlikely in ours)
            key = key.replace(",", " ")
            src = src.replace(",", " ")
            f.write(f"{idx},{key},{hp},{vp},{src}\n")
    print(f"[OK] CSV written:  {out_csv_path} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
