from tensorflow.keras import backend as K


def loss_function(y_true,y_pred):
    mask = K.cast(K.not_equal(y_true,0), K.floatx())
    squared_diff = K.square(y_true - y_pred)*mask
    mse = K.sum(squared_diff)/K.sum(mask)

    return mse