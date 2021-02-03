import os
import numpy as np
import soundfile as sf
import simpleaudio as sa
import time
import tensorflow.keras.backend as kb

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
    output_names = file_names[file_type != 'input']#
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
    
def InOut_plot(ax1,X,Y):
    color = 'tab:blue'
    ax1.plot(X, color = color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    color = 'tab:red'
    ax2 = ax1.twinx()
    ax2.plot(Y,color = color)
    ax2.tick_params(axis='y', labelcolor=color)
    
# def esr_dc(gamma = 1):
#     def loss(y_true,y_pred):
#         ypow = kb.mean(kb.square(y_true))
#         esr = kb.mean(kb.square(y_true - y_pred))/ypow
#         dc = kb.square(kb.mean(y_true - y_pred))/ypow
#         return gamma*esr+(1-gamma)*dc
#     return loss

def dc(y_true, y_pred):
    ypow = kb.mean(kb.square(y_true))
    return kb.square(kb.mean(y_true - y_pred))/(ypow+1e-5)


def esr(y_true,y_pred):
    ypow = kb.mean(kb.square(y_true))
    return kb.mean(kb.square(y_true - y_pred))/(ypow+1e-5)
   