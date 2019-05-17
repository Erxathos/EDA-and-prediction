import pandas as pd
def df_to_supervised(source_data, column_to_predict="cnt", lag=1, dropnan=True, reshape=True):
    """
    Returns a dataframe as a supervised learning framed dataset.
    Arguments:
        source_data: Observations as a DataFrame
        lag: Number of lag observations as input (X).
        dropnan: If true, drop rows with NaN values.
    """
    from pandas import concat
    # Empty dataframe to store the result
    x = pd.DataFrame()
    # Empty list to store new dataframe column names
    new_names = []
    
    # input sequence (t-n, ... t)
    for i in range(lag, 0, -1):
        # shift and put it all together
        x = concat([x, source_data.shift(i)], axis=1)
        # create a list of new variable names
        for col in source_data.columns:
            if i == 0:
                # first element (t)
                new_names += ['%s(t)' % col]
            else:
                # previous elements (t-1)
                new_names += ['%s(t-%d)' % (col, i)]
    # forecast sequence (t, t+1, ... t+n)
    y = source_data[column_to_predict].shift(-lag)
    
    # assign the new names to the resulting dataset
    x.columns = new_names
    
    # drop rows with NaN values
    if dropnan:
        x.dropna(inplace=True)
        x.reset_index(inplace=True, drop=True)
        y.dropna(inplace=True)
    
    # fit the size of x to the size of y
    x = x.iloc[0:len(y)]
    
    # Keras-LSTM-shaped
    # Non-reshaped can be used for pca/tsne data visualisation
    if (reshape):
        x = x.values.reshape(x.shape[0], (lag), x.shape[1]//(lag))
    return x, y

def load_data(lag=3*24, test_split=1/24):
    """
    Split the data into test and training datasets.
    Arguments:
        lag: lag variable to split the data
    """
    from sklearn.model_selection import train_test_split

    fname = "Dataset/hour.csv"
    df = pd.read_csv(fname, encoding='utf-8-sig')
    df.drop(["dteday", "instant"], axis=1,inplace=True)
    
    # Use the data of 3 previous days to predict the "cnt" of next hour
    x, y = df_to_supervised(df,lag=lag)
    # Roughly one month

    # We predict future, that is why we take the last part of data as a testing dataset (shuffle=False)
    x_train, x_test = train_test_split(x, test_size=test_split, shuffle=False)
    y_train, y_test = train_test_split(y, test_size=test_split, shuffle=False)
    y_test = y_test.values
    return x_train, y_train, x_test, y_test