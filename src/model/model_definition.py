import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers,optimizers

class DeepAutoEncoder:
    def __init__(self,input_shape,layers,activation,output_activation,dropout,regularization_encoder,regularization_decoder):
        
        self.input_shape = input_shape
        self.layers = layers
        self.activation = activation
        self.output_activation = output_activation
        self.dropout = dropout
        self.regularization_encoder = regularization_encoder
        self.regularization_decoder = regularization_decoder
        self.model = self.define_model()

    def define_model(self):

        x = input_layer = Input(shape=(self.input_shape,))

        ## Encoder
        k = int(len(self.layers)/2)

        for neurons in self.layers[:k]:
            x = Dense(neurons, activation=self.activation, kernel_regularizer=regularizers.l2(self.regularization_encoder))(x)

        ## Embedding
        x = Dense(self.layers[k], activation=self.activation, kernel_regularizer=regularizers.l2(self.regularization_encoder))(x)
        x = Dropout(rate=self.dropout)(x)

        ## Decoder
        for neurons in self.layers[k+1:]:
            x = Dense(neurons, activation=self.activation, kernel_regularizer=regularizers.l2(self.regularization_decoder))(x)

        output_layer = Dense(self.input_shape, activation=self.output_activation, kernel_regularizer=regularizers.l2(self.regularization_decoder))(x)

        model = Model(input_layer,output_layer)

        return model
    
    def get_model(self):
        return self.model
    
    def get_encoder_model(self):
        return Model(inputs=self.model.input, outputs=self.model.layers[int(len(self.layers)/2)].output)
