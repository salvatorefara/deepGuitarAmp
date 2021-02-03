
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import time, os, logging, sys

#os.environ['TF_CUDNN_RESET_RND_GEN_STATE'] = '1'

from utils import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as kb

# to stop tensorflow from printing messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# gpu memory management (attempt to allocate only as much memory as needed)
# check https://www.tensorflow.org/guide/gpu
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# gpu memory management (put a hard limit on the gpu memory available to tf)
# check https://www.tensorflow.org/guide/gpu
# limit = 1024*3 # memory limit (MB)
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(physical_devices[0], \
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit)])

#%% Load wave data

#data_dir = "D:/Dropbox/Projects/sound translator using gans/datasets/guitar_amplifier_dataset/wav/"
data_dir = "C:/Users/tore-/Dropbox/Projects/sound translator using gans/datasets/guitar_amplifier_dataset/wav/"
wavedata = load_wavedata(data_dir)
             
# check some shapes
# print(np.shape(wavedata['input']))
# print(np.shape(wavedata['markII']['gain6']))

#%% Plot wave data

idx = 21
samples = np.arange(0,int(2e5))

plt.figure()
plt.subplot(211)
plt.plot(wavedata['input'][idx][samples])
plt.subplot(212)
plt.plot(wavedata['markII']['gain6'][idx][samples])

#%% Play wave data

idx = 14
play_time = 2

audio_playback(wavedata['input'][idx], max_playback_time = play_time)
audio_playback(wavedata['markII']['gain6'][idx], max_playback_time = play_time)

#%% Format data (reshape into examples of length Tx)
# examples are overalapping by Trej, which are the amount of samples used to 
# initialize the state of the network, they are rejected before computing the 
# loss

Xwave = wavedata['input']
Ywave = wavedata['markII']['gain6']

# length of each training example (Tx), and length of transient (Trej)
Tx = round(10*wavedata['samplerate']) #2**12 #round(1*wavedata['samplerate'])
Trej = 2**10 #1000
Ttot = np.shape(Xwave)[1]
n_examples = int((Ttot-Tx)/(Tx-Trej)+1)
n_inputs = np.shape(Xwave)[0]

X = np.zeros((n_examples*np.shape(Xwave)[0],Tx))
Y = np.zeros((n_examples*np.shape(Xwave)[0],Tx))
for k in range(n_examples):
    example_start = k*(Tx-Trej)
    example_end = k*(Tx-Trej) + Tx
    X[k*n_inputs:(k+1)*n_inputs,:] = Xwave[:,example_start:example_end]
    Y[k*n_inputs:(k+1)*n_inputs,:] = Ywave[:,example_start:example_end]
    
#%% just a test to make sure the data examples overlap in the region to reject

k = 1000
assert(all(X[k,-Trej:-Trej+10]==X[k+n_inputs,:10]))
assert(all(Y[k,-Trej:-Trej+10]==Y[k+n_inputs,:10]))

#%% Plot a couple of examples

k = 50
samples = np.arange(0,1000)

plt.figure()
InOut_plot(plt.subplot(), X[k,samples], Y[k,samples])

#%% Play a couple of examples

k = 50
play_time = 2

audio_playback(X[k,:], max_playback_time = play_time)
audio_playback(Y[k,:], max_playback_time = play_time)

#%% Split into train, dev, test sets

tot_examples = np.shape(X)[0]
split_ratios = {'train':.8, 'dev':.1, 'test':.1}

# shuffle examples
idx_perm = np.random.RandomState(seed=1).permutation(tot_examples)

# train set 
sel_range = [0, int(tot_examples*split_ratios['train'])]
Xtrain = X[idx_perm[sel_range[0]:sel_range[1]],:]
Ytrain = Y[idx_perm[sel_range[0]:sel_range[1]],:]

# dev set
sel_range = [int(tot_examples*split_ratios['train']),  \
             int(tot_examples*(split_ratios['train']+split_ratios['dev']))]
Xdev = X[idx_perm[sel_range[0]:sel_range[1]],:]
Ydev = Y[idx_perm[sel_range[0]:sel_range[1]],:]

# test set
sel_range = [int(tot_examples*(split_ratios['train']+split_ratios['dev'])), \
              tot_examples]
Xtest = X[idx_perm[sel_range[0]:sel_range[1]],:]
Ytest = Y[idx_perm[sel_range[0]:sel_range[1]],:]

# make sure shapes match
assert(np.shape(Xtrain)[0]+np.shape(Xdev)[0]+np.shape(Xtest)[0] == tot_examples)
assert(np.shape(Ytrain)[0]+np.shape(Ydev)[0]+np.shape(Ytest)[0] == tot_examples)

print('# of training examples: %d' %(np.shape(Xtrain)[0]))
print('# of development examples: %d' %(np.shape(Xdev)[0]))
print('# of test examples: %d' %(np.shape(Xtest)[0]))

#%% Plot some examples

k = 10
samples = 2000

plt.figure()
InOut_plot(plt.subplot(131),Xtrain[k,:samples],Ytrain[k,:samples])
InOut_plot(plt.subplot(132),Xdev[k,:samples],Ydev[k,:samples])
InOut_plot(plt.subplot(133),Xtest[k,:samples],Ytest[k,:samples])

#%% Specify the RNN model

n_a = 32
initializer = keras.initializers.GlorotNormal()

# define global keras variables
LSTMcell = layers.GRU(units = n_a, return_sequences=True, kernel_initializer=initializer)
Densor = layers.Dense(1,activation='linear', kernel_initializer=initializer)
Preemphasis = layers.Conv1D(1,2,padding='valid',use_bias=False,trainable=False)
Preemphasis.build((1,))
Preemphasis.set_weights(np.array([-.85,1]).reshape((1,2,1,1)))

# to compensate for the "valid" convolution of the pre-emphasis filter
Trej_pe = Trej - 1

# Model
Xres = layers.Input(shape=(Tx,1))
X = LSTMcell(Xres)
X = Densor(X)
Y = layers.Add()([X,Xres])
Yrej = layers.Cropping1D((Trej_pe,0))(Y)
out_pe = layers.Cropping1D((Trej,0))(Y)
out = Preemphasis(Yrej)

model = keras.Model(inputs=[Xres], outputs=[out, out_pe])

print(model.summary())

#%% Compile model

learning_rate = .01

opt = tf.keras.optimizers.Adam(learning_rate,clipvalue=10)
#opt = tf.keras.optimizers.SGD(learning_rate,clipvalue=10)
model.compile(optimizer=opt, loss=[esr,dc], metrics=[esr,dc])
#model.compile(optimizer=opt, loss='mse', metrics=['mean_squared_error'])

#%% Show some baseline predictions

k = 10

with tf.device('/gpu:0'):
    y_hat = model.predict(Xtrain[k,:].reshape(1,Tx,1))

plt.figure()
plt.plot(Ytrain[k,:],color='b')
#plt.plot(y_hat[0],color='r')
plt.plot(np.arange(Trej,Tx),np.squeeze(y_hat[0]),color='r')
#plt.plot(Xtrain[k,:],color='k')

#%% Train the model

train_examples = np.arange(0,np.shape(Xtrain)[0])
#train_examples = np.arange(0,1500)
inputs = Xtrain[train_examples,:,np.newaxis]
outputs = Ytrain[train_examples,Trej:,np.newaxis]
inputs_dev = Xdev[:,:,np.newaxis]
outputs_dev = Ydev[:,Trej:,np.newaxis]

tic = time.process_time()
with tf.device('/gpu:0'):
    H = model.fit(inputs, outputs, epochs=10, batch_size=150, validation_data = (inputs_dev, outputs_dev))
toc = time.process_time()

print(toc - tic)

#%% Specify the RNN model (to hear the results)

n_a = 32
initializer = keras.initializers.GlorotNormal()
batch_size = 150 

# define global keras variables
LSTMcell = layers.GRU(units = n_a, return_sequences=True, kernel_initializer=initializer)
Densor = layers.Dense(1,activation='linear', kernel_initializer=initializer)
Preemphasis = layers.Conv1D(1,2,padding='valid',use_bias=False,trainable=False)
Preemphasis.build((1,))
Preemphasis.set_weights(np.array([-.85,1]).reshape((1,2,1,1)))

# to compensate for the "valid" convolution of the pre-emphasis filter
Trej_pe = Trej - 1

# Model
Xres = layers.Input(shape=(None,1))
X = LSTMcell(Xres)
X = Densor(X)
Y = layers.Add()([X,Xres])
#Yrej = layers.Cropping1D((Trej_pe,0))(Y)
out = Preemphasis(Y)

model = keras.Model(inputs=[Xres], outputs=[out])

print(model.summary())

#%% Compile model

learning_rate = .1

opt = tf.keras.optimizers.Adam(learning_rate,clipvalue=10)
model.compile(optimizer=opt, loss=[esr], metrics=[esr])

#%% Load trained weights

checkpoint_path = "C:/Users/tore-/Dropbox/Projects/sound translator using gans/model_checkpoint"
model.load_weights(checkpoint_path + '/gru_model_weights-50')

#%% Show some baseline predictions

k = 150

with tf.device('/cpu:0'):
    tic = time.process_time()
    y_hat = model.predict(X[k,:].reshape(1,np.shape(X)[1],1))
    toc = time.process_time()

print(toc - tic)

y_hat = np.squeeze(y_hat)

plt.figure()
plt.plot(Y[k,:],color='b')
plt.plot(y_hat[0],color='r')

play_time = 5
audio_playback(X[k,:], max_playback_time = play_time)
audio_playback(Y[k,:], max_playback_time = play_time)
audio_playback(y_hat[0], max_playback_time = play_time)
