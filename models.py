from utils import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def conv_model(nlayers=1, nchannels=8, kernel_size=2, dilation_rate=1, Tx=None, Trej=0, print_summary=False, learning_rate=.001, name='conv_model', loss='mse', metrics='mse', learn_difference=False, batchnorm=False, initializer = keras.initializers.GlorotNormal(), activation='relu'):
    '''
    
    Convolutional neural network model.
    
    '''
    
    # Single convolutional layer
    def conv_block(X, k, kernel_size, dilation_rate, batchnorm, activation):
        Z = layers.Conv1D(filters = nchannels,
                          kernel_size = kernel_size,
                          padding = 'causal',
                          dilation_rate = dilation_rate,
                          name = 'Z'+str(k), 
                          kernel_initializer=initializer)(X)
        if batchnorm is True:
            Z = layers.BatchNormalization(name='Z'+str(k)+'_norm')(Z)
        Z = layers.Activation(activation, name='Z'+str(k)+'_'+activation)(Z)
        Zmix = layers.Conv1D(1, 1, 
                             name = 'Z'+str(k)+'_mix', 
                             kernel_initializer=initializer)(Z)
        X = layers.Add(name = 'X'+str(k+1))([X,Zmix])
        return X, Z

    # Input layer
    X_input = layers.Input(shape = (Tx,1), name = 'X0')
    X = X_input

    # Hidden layers - stacked dilated convolutional layers
    Zlist = []
    for k in range(nlayers):
        X, Z = conv_block(X, k, kernel_size, int(dilation_rate[k]), batchnorm, activation)
        Zlist.append(Z)

    # Output layer
    Z = layers.Concatenate(axis=2, name='Z')(Zlist)
    Y = layers.Conv1D(1, 1, name = 'Y', kernel_initializer=initializer)(Z)
    
    # If we want the model to learn the difference between input and output (as done with the recurrent model)
    if learn_difference is True:
        Y = layers.Add(name='Ydiff')([Y,X_input])

    # Create model
    model = keras.Model(inputs=[X_input], outputs=Y, name=name)
    if print_summary:
        model.summary()

    # Compile model
    opt = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    
    return model

def recurrent_model(units=32, celltype='GRU', Tx=None, Trej=0, print_summary=False, learning_rate=.001, name='recurrent_model', loss='mse', metrics='mse', initializer = keras.initializers.GlorotNormal()):
    '''
    Recurrent neural network model. 
    
    '''
    
    # specify recurrent cell type
    if celltype=='GRU':
        Rcell = layers.GRU(units=units, return_sequences=True, kernel_initializer=initializer, name='Xrec')
    elif celltype=='LSTM':
        Rcell = layers.LSTM(units=units, return_sequences=True, kernel_initializer=initializer, name='Xrec')
    elif celltype=='RNN':
        Rcell = layers.SimpleRNN(units=units, return_sequences=True, kernel_initializer=initializer, name='Xrec')
    
    # Model architecture
    X_input = layers.Input(shape=(Tx,1), name='X0')
    X = Rcell(X_input)
    X = layers.Dense(1, activation='linear', kernel_initializer=initializer, name='Xdense')(X)
    Y = layers.Add(name='Y')([X,X_input])
    
    # Create model
    model = keras.Model(inputs=[X_input], outputs=Y, name=name)
    if print_summary:
        model.summary()
        
    # Compile model
    opt = tf.keras.optimizers.Adam(learning_rate,clipvalue=10)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    
    return model
