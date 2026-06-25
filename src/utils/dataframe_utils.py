import pandas as pd

#creation of the DataFrame
def  create_dataframe(csv_file: str, image_folder: str, segmented:bool=False):  
    """
    Load a CSV file containing image IDs and create a DataFrame
    with the corresponding image paths.
    ----------
    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing at least an 'id' column.

    image_folder : str
        Directory containing the images. If the path does not end
        with '/', it will be added automatically.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the original CSV columns and an
        additional 'path' column with the full path to each image.
    """
    if not image_folder.endswith("/"):
        image_folder += "/"
    df = pd.read_csv(csv_file)
    if segmented:
        df["path"]  = image_folder + df["id"].astype(str) + "_seg" + ".png"
    else: 
        df["path"]  = image_folder + df["id"].astype(str) + ".png"
    return df






