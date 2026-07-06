from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def scaling_data(df, df_val):
    """
    Scale the target variable (bone age) and create TensorFlow datasets
    for training and validation.

    The function extracts image paths and bone age labels from training
    and validation pandas DataFrames. The bone age values from the
    training set are standardized using StandardScaler. The same fitted
    scaler is then applied to the validation labels to ensure that both
    datasets are represented in the same normalized scale.

    TensorFlow Dataset objects are created containing pairs of image paths
    and scaled bone age values.

    Parameters
    ----------
    df : pandas.DataFrame
        Training dataframe containing at least two columns:
        - "path": path to the image file
        - "boneage": patient's bone age value

    df_val : pandas.DataFrame
        Validation dataframe containing at least two columns:
        - "path": path to the image file
        - "boneage": patient's bone age value

    Returns
    -------
    dataset : tf.data.Dataset
        TensorFlow dataset for training containing tuples:
        (image_path, scaled_bone_age)

    dataset_val : tf.data.Dataset
        TensorFlow dataset for validation containing tuples:
        (image_path, scaled_bone_age)

    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler object using only the training labels.
        It can be used to inverse-transform model predictions
        from normalized values back to the original bone age scale.
    """
    paths = df["path"].values
    labels = df[["boneage"]].values.astype("float32")
    paths_val = df_val["path"].values
    labels_val = df_val[["boneage"]].values.astype("float32")

    scaler =StandardScaler()
    scaler.fit(labels)

    labels_scaled = scaler.transform(labels)
    labels_val_scaled = scaler.transform(labels_val)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels_scaled))
    dataset_val = tf.data.Dataset.from_tensor_slices((paths_val, labels_val_scaled))
    return dataset, dataset_val, scaler



def scaling_data_torch(df, df_val, df_test):
    """
    Scale the target variable (bone age) using StandardScaler for training, 
    validation and test datasets.

    The function extracts the bone age labels from the provided training,
    validation and test pandas DataFrames. A StandardScaler is fitted using
    only the training labels to compute the mean and standard deviation.
    The fitted scaler is then applied to training, validation and test
    labels to ensure consistent normalization across datasets.

    The scaled bone age values replace the original "boneage" column
    in both DataFrames. The fitted scaler is returned so that predictions
    produced by a PyTorch model can later be inverse-transformed back
    to the original bone age scale.

    Parameters
    ----------
    df : pandas.DataFrame
        Training dataframe containing at least the column:
        - "boneage": patient's bone age value

    df_val : pandas.DataFrame
        Validation dataframe containing at least the column:
        - "boneage": patient's bone age value

    Returns
    -------
    df : pandas.DataFrame
        Training dataframe with the "boneage" column replaced by
        standardized values.

    df_val : pandas.DataFrame
        Validation dataframe with the "boneage" column replaced by
        standardized values.

    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler object using only the training labels.
        Useful for inverse-transforming model predictions back to
        the original bone age scale.
    """
    labels = df[["boneage"]].values.astype("float32")
    labels_val = df_val[["boneage"]].values.astype("float32")
    labels_test = df_test[["boneage"]].values.astype("float32")

    scaler = StandardScaler()
    scaler.fit(labels)

    df[["boneage"]] = scaler.transform(labels)
    df_val[["boneage"]]= scaler.transform(labels_val)
    df_test[["boneage"]]= scaler.transform(labels_test)

    return df, df_val, df_test, scaler