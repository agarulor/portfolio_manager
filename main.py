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






    #g = calculate_covariance(train_set)

#h = create_markowitz_table(train_set, test_set, g, rf = 0.000, min_w=0.025, max_w=0.16, custom_target_volatility=0.26)

#print(h)
    run_app()
#plot_frontier(60, train_set, g,  method="simple", min_w=0.025, max_w=0.20, custom_target_volatility=0.26)


if __name__ == "__main__":
    main()