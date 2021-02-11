import os
import numpy as np
import soundfile as sf
import simpleaudio as sa
import time
import tensorflow.keras.backend as kb
import tensorflow as tf
from IPython.display import display, Audio

def round_nextpow2(v):
    '''
    Round up to the next highest power of 2. 
    Source:
    https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    
    '''
    
    v = int(v)
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v += 1
    v += v==0
    
    return v

def load_wavedata(data_dir):
    """
    Parameters
    ----------
    data_dir : str
        Directory where wave files are stored.

    Returns
    -------
    wavedata : dict
        Dictionary containing the loaded wave files.

    """
    # retrieve all wave file names from data folder
    file_names = os.listdir(data_dir)
    if file_names.count('README.txt'): file_names.remove('README.txt')
    file_names = np.array(file_names)
    
    # Retreive input and output file names
    file_type = np.array([x.split('-')[0] for x in file_names])
    input_names = file_names[file_type == 'input']
    output_names = file_names[file_type != 'input']
    n_inputs = len(input_names)
    
    # Load data into a dictionary (parse amp types and specs)
    wavedata = {}
    for input_idx,input_name in enumerate(input_names):
        # Load input
        input_file,wavedata['samplerate'] = sf.read(data_dir + input_name)
        
        # stack input data
        if not ('input' in wavedata):
            wavedata['input'] = np.zeros((n_inputs,len(input_file)))
        wavedata['input'][input_idx,:] = input_file
        
        # find corresponding output names
        curr_output_names = output_names[[x.count(input_name.split('-',1)[1])>0 for x in output_names]]
        
        # get amps and specs
        curr_amps = np.array([x.split('-')[0] for x in curr_output_names])
        curr_specs = np.array([x.split('-',2)[1] for x in curr_output_names])
        for output_idx,output_name in enumerate(curr_output_names):
            output_file, _ = sf.read(data_dir + output_name)
            
            if not (curr_amps[output_idx] in wavedata):
                wavedata[curr_amps[output_idx]] = \
                    {curr_specs[output_idx]: np.zeros((n_inputs,len(input_file)))}
            else:
                if not (curr_specs[output_idx] in wavedata[curr_amps[output_idx]]):
                    wavedata[curr_amps[output_idx]][curr_specs[output_idx]] = \
                        np.zeros((n_inputs,len(input_file)))
            wavedata[curr_amps[output_idx]][curr_specs[output_idx]][input_idx,:] = \
                output_file
            #print(output_idx)
            #print(curr_amps[output_idx])
            #print(curr_specs[output_idx])
            #print(wavedata)
    
    return wavedata

def data_chunk(X, Tx, Trej = 0):
    '''
    
    Chunks X into examples of length Tx, with overlap Trej. 
    
    Parameters
    ----------
    X : numpy.ndarray
        Data to be chunked, with shape = (examples,time)
    Tx: int
        Chunk size
    Trej: int, optional
        Chunk overlap; default is 0

    Returns
    -------
    Xchunk : numpy.ndarray
        Chunked array.

    '''
    
    Ttot = np.shape(X)[1]
    n_examples = int((Ttot-Tx)/(Tx-Trej)+1)
    n_inputs = np.shape(X)[0]

    Xchunk = np.zeros((n_examples*np.shape(X)[0],Tx))
    for k in range(n_examples):
        example_start = k*(Tx-Trej)
        example_end = k*(Tx-Trej) + Tx
        Xchunk[k*n_inputs:(k+1)*n_inputs,:] = X[:,example_start:example_end]

    # just a test to make sure the data examples overlap in the region to reject
    k = int(np.shape(X)[0]/2)
    if Trej:
        assert(all(Xchunk[k,-Trej:]==Xchunk[k+n_inputs,:Trej]))
        
    return Xchunk

def audio_playback(audio_data, samplerate = 44100, max_playback_time = 2):
    '''
    
    Parameters
    ----------
    audio_data : str or numpy.ndarray
        Either a file name or a numpy array containing wave data 
    samplerate : int, optional
        The default is 44100.
    max_playback_time : int, optional
        The default is 2.

    Returns
    -------
    None.

    '''
    file_name = 'audio_playback.wav'
    
    # if the audio data is a numpy array, create a corresponding audio file
    if isinstance(audio_data, np.ndarray):
        # make sure audio data has shape = frames*channels
        if len(np.shape(audio_data))>1:
            # here we assume nframes>2 
            if np.shape(audio_data)[0] < np.shape(audio_data)[1]:
                audio_data = audio_data.T
        sf.write(file_name,audio_data,samplerate)
    
    # play audio for max_playback_time seconds    
    wave_obj = sa.WaveObject.from_wave_file(file_name)
    play_obj = wave_obj.play()
    time.sleep(max_playback_time)
    play_obj.stop()
    
def audio_playback_gui(audio_data, samplerate = 44100, file_name = 'audio_example_0', audio_example_folder = 'audio_examples'):
    '''
    Improved version of audio_playback() based on IPython; it integrates well
    with jupyter notebooks. 
    
    Parameters
    ----------
    audio_data : str or numpy.ndarray
        Either a file name or a numpy array containing wave data 
    samplerate : int, optional
        The default is 44100.
        Unused if audio_data is string.
    file_name : str, optional
        The default is 'audio_example_0'.
        Unused if audio_data is string.
    audio_example_path: str, optional
        The default is 'audio_examples'.
        Unused if audio_data is string.

    Returns
    -------
    None.

    '''

    # if audio_data is a numpy array, create a corresponding audio file
    if isinstance(audio_data, np.ndarray):

        # create a folder to store the audio examples (if it does not exist yet)
        if not os.path.exists(audio_example_folder):
            os.makedirs(audio_example_folder)

        # make sure audio data has shape = frames*channels
        if len(np.shape(audio_data))>1:
            # here we assume nframes>2 
            if np.shape(audio_data)[0] < np.shape(audio_data)[1]:
                audio_data = audio_data.T

        # create audio file
        file_path = audio_example_folder + '/' + file_name + '.wav'
        sf.write(file_path,audio_data,samplerate)

    else:
        file_path = audio_data

    # play audio
    display(Audio(file_path))
    
def InOut_plot(ax1,X,Y):
    color = 'tab:blue'
    ax1.plot(X, color = color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    color = 'tab:red'
    ax2 = ax1.twinx()
    ax2.plot(Y,color = color)
    ax2.tick_params(axis='y', labelcolor=color)

def dc(y_true, y_pred):
    '''
    
    DC loss.
    
    '''
    
    ypow = kb.mean(kb.square(y_true))
    return kb.square(kb.mean(y_true - y_pred))/(ypow+1e-10)

def esr(y_true,y_pred):
    '''
    
    Error-to-signal ratio (esr).
    
    '''
    
    ypow = kb.mean(kb.square(y_true))
    return kb.mean(kb.square(y_true - y_pred))/(ypow+1e-10)

def esr_dc(Trej=None, preemphasis=None):
    '''
    
    Error-to-signal ratio (esr) + DC loss.
    
    '''
    
    def loss(y_true,y_pred):
        # Reject model transients
        if Trej is not None:
            y_true = y_true[:,Trej:,:]
            y_pred = y_pred[:,Trej:,:]
        
        # Compute dc loss
        dc_loss = dc(y_true, y_pred)
        
        # Apply pre-emphasis filter
        if preemphasis is not None:
            y_true = tf.nn.conv1d(y_true, preemphasis, stride=1, padding='SAME')
            y_pred = tf.nn.conv1d(y_pred, preemphasis, stride=1, padding='SAME')
        
        ypow = kb.mean(kb.square(y_true))
        return kb.mean(kb.square(y_true - y_pred))/(ypow+1e-10) + dc_loss
    return loss