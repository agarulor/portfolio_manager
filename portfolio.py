import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from interface.run_app import run_app
import os
import random
import numpy as np

SEED = 42
# To avoid changes in hash
os.environ["PYTHONHASHSEED"] = str(SEED)

# Seeds for Python and NumPy
random.seed(SEED)
np.random.seed(SEED)


def main():
    """
    Computes main.

    Parameters
    ----------

    Returns
    -------
    Any: main output.
    """
    run_app()


if __name__ == "__main__":
    main()