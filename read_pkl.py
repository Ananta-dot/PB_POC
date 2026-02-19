import pickle
import pprint

pkl_path = "runs/misr_run-2025-10-12_10-19-37-seed123/n14/round10_elites_n14_2025-10-12_11-17-53.pkl"

with open(pkl_path, "rb") as f:
    obj = pickle.load(f)

pprint.pprint(obj, width=120)
