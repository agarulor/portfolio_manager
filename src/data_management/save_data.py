import os
import pandas as pd
from datetime import datetime

def save_preprocessed_data(df: pd.DataFrame,
                           save_path: str = "data/processed",
                           file_prefix: str = "prices") -> None:
    """
    Save the preprocessed price dataset to a csv file with a timestamp

    Saves the  DataFrame into a csv file located in the specified directory.
    The file name includes a timestamp to avoid deletions of previous files

    Parameters
    ----------
    df : pandas.DataFrame
        The preprocessed price dataset
    save_path : str, default "data/processed"
        Directory where we are saving our csv. If it doesn't exist it will be created automatically
    file_prefix : str, default "prices"
        Name of the ouptut file

    Returns
    -------
    None
        It saves the dataset to a directory in the system
    """

    # We first check that the directory exists
    os.makedirs(save_path, exist_ok=True)

    # We create the file with the timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # We generate the file_name
    file_name = f"{file_prefix}_{timestamp}.csv"

    # Now we generate the save_path + file_name
    file_path = os.path.join(save_path, file_name)

    # Finally we save the DataFrame
    df.to_csv(file_path, index_label=False)

    print(f"Saved preprocessed data to {file_path}")