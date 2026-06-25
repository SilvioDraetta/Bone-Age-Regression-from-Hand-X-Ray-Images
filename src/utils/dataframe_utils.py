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

#Function for the merge of datasets

def merge_dataset(df, df_segmented):
    """
    Merge two pandas DataFrames by concatenating their rows.

    The function combines the original dataset with a second dataset
    (e.g. a segmented version of the images) by appending the rows of
    the second dataframe below the first one.

    Parameters
    ----------
    df : pandas.DataFrame
        Original dataframe containing the dataset samples.

    df_segmented : pandas.DataFrame
        Dataframe containing additional samples to be added.
        It should have the same column structure as the first one.

    Returns
    -------
    pandas.DataFrame
        A new dataframe containing all samples from both input dataframes,
        with a reset continuous index.
    """
    df_all = pd.concat([df, df_segmented], ignore_index=True)
    return df_all






