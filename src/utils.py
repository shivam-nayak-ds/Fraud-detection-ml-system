# src/utils.py

import pickle
import os


# ── 1. Save Pickle ────────────────────────────────────────────────────────────
def save_pickle(obj, path):
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved: {path}")


# ── 2. Load Pickle ────────────────────────────────────────────────────────────
def load_pickle(path):
   
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    print(f"Loaded: {path}")
    return obj


# ── 3. Create Directories ─────────────────────────────────────────────────────
def create_dirs(*paths):
    
    for path in paths:
        os.makedirs(path, exist_ok=True)
        print(f"Directory ready: {path}")