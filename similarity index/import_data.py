import scipy.io
import pandas as pd
import numpy as np
import os
import nelpy as nel


def load_cell_metrics(project_path: str, mice_name : str) -> tuple:

    mat_data_cell = scipy.io.loadmat(project_path+ '/'+ mice_name+'/'+'cell_id_region.mat')
    mat_data_behav = scipy.io.loadmat(project_path+ '/'+ mice_name+'/'+'behavior_times.mat')
    mat_data_spk = scipy.io.loadmat(project_path+ '/'+ mice_name+'/'+mice_name.split('_')[1]+'.spikes.cellinfo.mat')
    mat_data_swr = scipy.io.loadmat(project_path+ '/'+ mice_name+'/'+'SWR.mat')
    for file_name in os.listdir(project_path+ '/'+ mice_name):
        if "Sleep" in file_name:
            mat_data_sleep= scipy.io.loadmat(project_path+ '/'+ mice_name+'/'+file_name)
        
    
    try: 
        for key,data in mat_data_sleep.items():
            if "Sleep" in key:
                dtype_names = mat_data_sleep[key]["ints"][0][0][0].dtype.names   
                wake_id= dtype_names.index("WAKEepisode")+1
                rem_id  = dtype_names.index("REMepisode")+1
                nrem_id  = dtype_names.index("NREMepisode")+1
    
                # set up dict
                dict_ = {
                    "wake_id": wake_id,
                    "rem_id": rem_id,
                    "nrem_id": nrem_id,
                }
    
                # iter through states and add to dict
                dt = mat_data_sleep[key]["ints"][0][0].dtype
                for dn in dt.names:
                    dict_[dn] =  mat_data_sleep[key]["ints"][0][0][dn][0][0]
    except:
            
        for key,data in mat_data_sleep.items():
            if "Sleep" in key:
                dtype_names = mat_data_sleep[key]["ints"][0][0][0].dtype.names   
                wake_id= dtype_names.index("WAKEstate")+1
                rem_id  = dtype_names.index("REMstate")+1
                nrem_id  = dtype_names.index("NREMstate")+1
        
                # set up dict
                dict_ = {
                    "wake_id": wake_id,
                    "rem_id": rem_id,
                    "nrem_id": nrem_id,
                }
        
                # iter through states and add to dict
                dt = mat_data_sleep[key]["ints"][0][0].dtype
                for dn in dt.names:
                    dict_[dn] =  mat_data_sleep[key]["ints"][0][0][dn][0][0]



    '''cell info'''
    cell_metrics_structure = mat_data_cell['cell_id_region']
    
    df_cell =  pd.DataFrame(cell_metrics_structure, columns=['Type', 'Region', 'Modulation'])
    
    type_map = {1: 'pyr', 2: 'int'}
    region_map = {1: 'CA1', 2: 'CA2', 3: 'CA3'}
    modulation_map = {1: 'positive', 2: 'negative'}
    
    df_cell['Type'] = df_cell['Type'].map(type_map)
    df_cell['Region'] = df_cell['Region'].map(region_map)
    df_cell['Modulation'] = df_cell['Modulation'].map(modulation_map)
    df_cell['CluID'] = list(range(0,len(df_cell)))

    
    ''''behv'''
    
    behavior_times = mat_data_behav['behavior_times'][0, 0].flatten()
    
    cage_or_maze = []
    start_time = []
    end_time = []
    
    
    for field_name in behavior_times.dtype.names:
        data = behavior_times[field_name][0]
        if data.size > 0:
            cage_or_maze.append(field_name)
            start_time.append(data[0][0])
            end_time.append(data[0][1])
    df_behav = pd.DataFrame({'Cage_or_Maze': cage_or_maze, 'Start_Time': start_time, 'End_Time': end_time})
    
    behav_map = {'cage1':'rest_hab_pre',
                 'maze1':'habituation_arena',
                 'cage2':'rest_hab_post',
                 'maze2':'habituation_cage',
                 'cage3':'rest_pre',
                 'maze3':'2novel_exposure',
                 'maze4':'exposure_reversed',
                 'cage4':'rest_post2',
                 'maze5':'1novel_exposure',
                 'cage5':'rest_post1'}
    
    df_behav['Cage_or_Maze'] = df_behav['Cage_or_Maze'].map(behav_map) # >>>> epochs!

    '''SPIKES'''
    spikes = [
        spk.T[0] for spk in mat_data_spk["spikes"][0][0]["times"][0]
    ]
    
    
    
    '''SWR'''
    df_swr = pd.DataFrame()
    df_swr['peaktimes']=mat_data_swr['SWR']['peaktimes'][0][0][:,0]
    df_swr['start']= mat_data_swr['SWR']['timestamps'][0][0][:, 0]
    df_swr['stop']= mat_data_swr['SWR']['timestamps'][0][0][:, 1]

    return df_cell, df_behav, spikes, df_swr, dict_



def load_epochs(project_path: str, mice_name : str):
    _,epochs,_,_,_ = load_cell_metrics(project_path, mice_name)
    return epochs

def load_ripples_events(project_path: str, mice_name : str):
    _,_,_,SWR,_ = load_cell_metrics(project_path, mice_name)
    return SWR

def load_SleepState_states(project_path: str, mice_name : str):
    _,_,_,_,BrainState = load_cell_metrics(project_path, mice_name)
    return BrainState



def load_spikes(
    project_path,
    mice_name,
    putativeCellType=[],  # restrict spikes to putativeCellType
    brainRegion=[],  # restrict spikes to brainRegion
    brain_state=[],  # restrict spikes to brainstate
    support=None,  # provide time support
):
    

    fs_dat= 30000
    if not isinstance(putativeCellType, list):
        putativeCellType = [putativeCellType]
    if not isinstance(brainRegion, list):
        brainRegion = [brainRegion]

    # load cell metrics and spike data
    cell_metrics,epochs, data, swr, brainstate = load_cell_metrics(project_path, mice_name)

    if cell_metrics is None or data is None:
        return None, None

    # put spike data into array st
    st = np.array(data, dtype=object)

    # restrict cell metrics
    if len(putativeCellType) > 0:
        restrict_idx = []
        for cell_type in putativeCellType:
            restrict_idx.append(
               cell_metrics['Type'].str.contains(cell_type).values
            )
        restrict_idx = np.any(restrict_idx, axis=0)
        cell_metrics = cell_metrics[restrict_idx]
        st = st[restrict_idx]

    

    if len(brainRegion) > 0:
        restrict_idx = []
        
        for brain_region in brainRegion:
            if len(brain_region)==3:
                    
                restrict_idx.append(
                    cell_metrics['Region'].str.contains(brain_region).values
                )
                
                restrict_idx = np.any(restrict_idx, axis=0)
                cell_metrics = cell_metrics[restrict_idx]
                st = st[restrict_idx]
                
                
            elif len(brain_region) > 4:
                brain_regions= brain_region.split('-')
                sts=[]
                cell_dfs= []
                for b_region in brain_regions:
                    
                    cell_metrics_regions= cell_metrics
                    st_regions= st
                    restrict_idx = []

                    restrict_idx.append(
                        cell_metrics_regions['Region'].str.contains(b_region).values
                    )
                    
                    restrict_idx = np.any(restrict_idx, axis=0)
                    cell_metrics_regions = cell_metrics_regions[restrict_idx]
                    st_region = st_regions[restrict_idx]
                    sts.append(st_region)
                    cell_dfs.append(cell_metrics_regions)
                st = np.concatenate((sts[0],sts[1]))
                cell_metrics= pd.concat([cell_dfs[0],cell_dfs[1]])

  
    
    max_length = max(len(arr) for arr in st)
    result_array = np.zeros(( max_length,st.size))
    
    for i, arr in enumerate(st):
        result_array[:len(arr), i] = arr
    st= result_array.T

    try:
        if support is not None:
            st = nel.SpikeTrainArray(timestamps= st, fs=fs_dat, support=support)
        else:
            st = nel.SpikeTrainArray(timestamps=st, fs=fs_dat)
    except:  # if only single cell... should prob just skip session
        if support is not None:
            st = nel.SpikeTrainArray(timestamps=st[0], fs=fs_dat, support=support)
        else:
            st = nel.SpikeTrainArray(timestamps=st[0], fs=fs_dat)

    if len(brain_state) > 0:
        # get brain states
        brain_states = ["WAKEepisode", "NREMepisode", "REMepisode"]
        if brain_state not in brain_states:
            assert print("not correct brain state. Pick one", brain_states)
        else:
            state_dict = brainstate
            state_epoch = nel.EpochArray(state_dict[brain_state])
            st = st[state_epoch]

    return st, cell_metrics




def event_triggered_average_fast(
    signal: np.ndarray,
    events: np.ndarray,
    sampling_rate: int,
    window=[-0.5, 0.5],
    return_average: bool = True,
    return_pandas: bool = False,
):
    """
    event_triggered_average: Calculate the event triggered average of a signal

    Args:
        signal (np.ndarray): 2D array of signal data (channels x timepoints)
        events (np.ndarray): 1D array of event times
        sampling_rate (int): Sampling rate of signal.
        window (list, optional): Time window (seconds) to average signal around event. Defaults to [-0.5, 0.5].
        return_average (bool, optional): Whether to return the average of the event triggered average. Defaults to True.
            if False, returns the full event triggered average matrix (channels x timepoints x events)

    Returns:
        np.ndarray: Event triggered average of signal
        np.ndarray: Time lags of event triggered average

    note: This version assumes constant sampling rate, no missing data (time gaps), signal start time at 0
    """

    window_starttime, window_stoptime = window
    window_bins = int(np.ceil(((window_stoptime - window_starttime) * sampling_rate)))
    time_lags = np.linspace(window_starttime, window_stoptime, window_bins)

    events = events[
        (events * sampling_rate > len(time_lags) / 2 + 1)
        & (events * sampling_rate < signal.shape[1] - len(time_lags) / 2 + 1)
    ]

    avg_signal = np.zeros(
        [signal.shape[0], len(time_lags), len(events)], dtype=signal.dtype
    )

    for i, event in enumerate(events):
        ts_idx = np.arange(
            np.round(event * sampling_rate) - len(time_lags) / 2,
            np.round(event * sampling_rate) + len(time_lags) / 2,
        ).astype(int)
        avg_signal[:, :, i] = signal[:, ts_idx]

    if return_pandas and return_average:
        return pd.DataFrame(
            index=time_lags,
            columns=np.arange(signal.shape[0]),
            data=avg_signal.mean(axis=2).T,
        )
    if return_average:
        return avg_signal.mean(axis=2), time_lags
    else:
        return avg_signal, time_lags
