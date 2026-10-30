"""
Comprehensive builder and executor script for the Flock Energy Data Science Project.
Builds and executes all 8 notebooks according to the exact requested folder structure.
"""

import os
import sys
import json
import time
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

PROJECT_ROOT = Path(".").resolve()

def make_kernel_spec():
    return {
        "language_info": {"name": "python", "version": "3.13.7"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    }

def execute_notebook(nb_path: Path):
    print(f"\n=======================================================")
    print(f"Executing: {nb_path}")
    print(f"Working Directory: {nb_path.parent}")
    print(f"=======================================================")
    t0 = time.time()
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=900, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})

    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Finished {nb_path.name} in {time.time()-t0:.1f}s!\n")

print("Builder script ready.")
