import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from app.run_app import run_app


import os
import random
import numpy as np

SEED = 42

# Para que el hashing de Python no cambie entre ejecuciones
os.environ["PYTHONHASHSEED"] = str(SEED)

# Semillas de Python, NumPy y TensorFlow
random.seed(SEED)
np.random.seed(SEED)


def main():
    run_app()

if __name__ == "__main__":
    main()