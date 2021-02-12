from utils import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def conv_model(nlayers=10, nchannels=10, Tx=None, Trej=0, print_summary=False, learning_rate=.001, name='conv_model', loss='mse', metrics=['mse']):
    '''
    
    Convolutional neural network model.
    
    '''
    
    # Xavier normal inizializer for the weights
    initializer = keras.initializers.GlorotNormal()
    
    # Single convolutional layer
    def conv_block(X,k,dilation):
        Z = layers.Conv1D(filters = nchannels,
                          kernel_size = 2,
                          padding = 'causal',
                          dilation_rate = dilation,
                          activation = 'relu',
                          name = 'Z'+str(k), 
                          kernel_initializer=initializer)(X)
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
        X, Z = conv_block(X,k,dilation=2**k)
        Zlist.append(Z)

    # Output layer
    Z = layers.Concatenate(axis=2, name='Z')(Zlist)
    Y = layers.Conv1D(1, 1, name = 'Y', kernel_initializer=initializer)(Z)

    # Create model
    model = keras.Model(inputs=[X_input], outputs=Y, name=name)
    if print_summary:
        model.summary()

    # Compile model
    opt = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    
    return model

def recurrent_model(units=32, celltype='GRU', Tx=None, Trej=0, print_summary=False, learning_rate=.001, name='recurrent_model', loss='mse', metrics=['mse']):
    '''
    Recurrent neural network model. 
    
    '''
    
    # Xavier normal inizializer for the weights
    initializer = keras.initializers.GlorotNormal()

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

def recurrent_model_old(units=32, celltype='GRU', Tx=None, Trej=0, preemphasis=True, preemphasis_weights=[-.95,1], print_summary=False, learning_rate=.001):
    '''
    Recurrent neural network model with or without pre-emphasis filter. 
    
    '''
    
    # Xavier normal inizializer for the weights
    initializer = keras.initializers.GlorotNormal()

    # specify recurrent cell type
    if celltype=='GRU':
        Rcell = layers.GRU(units=units, return_sequences=True, kernel_initializer=initializer)
    elif celltype=='LSTM':
        Rcell = layers.LSTM(units=units, return_sequences=True, kernel_initializer=initializer)
    elif celltype=='RNN':
        Rcell = layers.SimpleRNN(units=units, return_sequences=True, kernel_initializer=initializer)
    
    # specify preemphasis filter 
    if preemphasis:
        Preemphasis = layers.Conv1D(1,2,padding='valid',use_bias=False,trainable=False, \
                                   name = 'preemphasis')
        Preemphasis.build((1,))
        Preemphasis.set_weights(np.array(preemphasis_weights).reshape((1,2,1,1)))
    
    # Base model
    Xres = layers.Input(shape=(Tx,1))
    X = Rcell(Xres)
    X = layers.Dense(1, activation='linear', kernel_initializer=initializer)(X)
    Y = layers.Add()([X,Xres])
    
    # Set the output of the model:
    # - during training (Trej>0), add a cropping layer to remove model trainsients
    # - add preemphasis filter if needed 
    if Trej==0 and not preemphasis:
        outputs = Y
    elif Trej==0 and preemphasis:
        outputs = Preemphasis(Y)
    elif Trej and not preemphasis:
        outputs = layers.Cropping1D((Trej,0))(Y)
    elif Trej and preemphasis:
        out = layers.Cropping1D((Trej,0), name = 'dc')(Y)
        
        # crop less to compensate for the "valid" convolution of the pre-
        # emphasis filter
        Ype = layers.Cropping1D((Trej-1,0))(Y)
        out_pe = Preemphasis(Ype)
        
        # return two outputs: after (out_pe) and before (out) the pre-emphasis
        # filter
        outputs = [out_pe, out]
    
    # create the model
    model = keras.Model(inputs=[Xres], outputs=outputs)
    if print_summary:
        print(model.summary())
        
    # compile model
    opt = tf.keras.optimizers.Adam(learning_rate,clipvalue=10)
    if Trej and preemphasis:
        loss = [esr,dc]
        metrics = [esr,dc]
    else:
        loss = esr
        metrics = esr
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    
    return model