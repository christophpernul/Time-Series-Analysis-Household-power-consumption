import numpy as np

def scaler_transform_1D(scaler, dataframe, fit=False):
    """
    Transforms input data using the given scaler in the case of one-dimensional input (shape: (n,)) by
    transforming it to shape (n,1), which is the needed shape for the scaler object.
    :param scaler: pandas scaler object
    :var dataframe: dataframe, which should be scaled (shape (n,))
    :param fit: boolean variable that decides whether there should be an additional .fit() using the scaler object
    :return: returns array of type (n,)
    """
    if np.shape(dataframe)[1] != 1:
        raise Exception(
            'Error: Got {}-dim data: Use this function only for one-dimensional input data!\n'.format(np.shape(dataframe)) +
            'For multi-dimensional data use built in sklearn.preprocessing transform attribute of specified scaler '
               'object!')
    else:
        if fit == True:
            return np.squeeze(scaler.fit_transform(np.array(dataframe).reshape(-1,1)))
        elif not fit:
            return np.squeeze(scaler.transform(np.array(dataframe).reshape(-1,1)))

def scaler_inverse_transform_1D(scaler, dataframe):
    """
        Performs Inverse transformation of input data using the given scaler in the case of one-dimensional input (shape: (n,)) by
        transforming it to shape (n,1), which is the needed shape for the scaler object.
        :param scaler: pandas scaler object
        :var dataframe: dataframe, which should be scaled (shape (n,))
        :return: returns array of type (n,)
        """
    return np.squeeze(scaler.inverse_transform(np.array(dataframe).reshape(-1,1)))

def df_get_num_cat(dataframe):
    """Splits dataframe into numeric and categorical columns and returns a tuple
        consisting of a list of the numeric column names and a list containing the name of categorical columns
        :var dataframe: pandas dataframe object
        :return: tupel of lists (first list: includes names of numerical columns, second list: names of categorical columns)
    """
    cols = dataframe.columns.values
    kinds = np.array([dt.kind for dt in dataframe.dtypes])
    is_num = kinds != 'O'
    num_cols = cols[is_num]
    cat_cols = cols[~is_num]
    return (num_cols, cat_cols)