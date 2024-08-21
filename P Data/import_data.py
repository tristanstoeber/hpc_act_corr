import scipy.io
import pandas as pd
import numpy as np
import os
import nelpy as nel
import hdf5storage


def load_cell_metrics(project_path: str, mice_name : str) -> tuple:
    try:
        mat_data_cell = scipy.io.loadmat(project_path+ '/'+ mice_name+'/'+mice_name+'.cell_metrics.cellinfo.mat')
        mat_data_ses = scipy.io.loadmat(project_path+ '/'+ mice_name+'/'+mice_name+'.session.mat')
        
    except:
        mat_data_cell = hdf5storage.loadmat(project_path+ '/'+ mice_name+'/'+mice_name+'.cell_metrics.cellinfo.mat')
        mat_data_ses = hdf5storage.loadmat(project_path+ '/'+ mice_name+'/'+mice_name+'.session.mat')

   

    '''cell info'''
    
    cell_metrics_structure = mat_data_cell['cell_metrics']
    brain_regions = [region[0] for region in cell_metrics_structure['brainRegion'][0,0].flatten()]
    cell_type = [cell[0] for cell in cell_metrics_structure['putativeCellType'][0,0].flatten()]

    df_cell = pd.DataFrame({'BrainRegion': brain_regions,'CellType':cell_type})
    df_cell['CluID'] = list(range(0,len(df_cell)))

   

    info_session = mat_data_ses['session']

    info_epochs = info_session['epochs'][0, 0].flatten()

   
    info_epochs_data = []
    intersleep= []
    for epoch in info_epochs:
        behavioralParadigm = epoch['behavioralParadigm'][0][0][0] if isinstance(epoch['behavioralParadigm'][0][0], np.ndarray) else epoch['behavioralParadigm'][0][0]
        
        if behavioralParadigm == 'InterSleep1':
            epoch_start = epoch['startTime'][0][0][0][0] if isinstance(epoch['startTime'][0][0][0], np.ndarray) else epoch['startTime'][0][0][0]
            intersleep.append(epoch_start)
        
        elif behavioralParadigm == 'InterSleep2':
            epoch_end = epoch['stopTime'][0][0][0][0] if isinstance(epoch['stopTime'][0][0][0], np.ndarray) else epoch['stopTime'][0][0][0]
            intersleep.append(epoch_end)
            info_epochs_data.append({'BehavioralParadigm': 'InterSleep',
                                     'Start_Time': intersleep[0],
                                     'End_Time': intersleep[1]})

            
        else:
            epoch_start = epoch['startTime'][0][0][0][0] if isinstance(epoch['startTime'][0][0][0], np.ndarray) else epoch['startTime'][0][0][0]
            epoch_end = epoch['stopTime'][0][0][0][0] if isinstance(epoch['stopTime'][0][0][0], np.ndarray) else epoch['stopTime'][0][0][0]
            info_epochs_data.append({'BehavioralParadigm': behavioralParadigm,
                                     'Start_Time': epoch_start,
                                     'End_Time': epoch_end})


    df_behav = pd.DataFrame(info_epochs_data)

    '''SPIKES'''
    spikes = cell_metrics_structure[0, 0]['spikes'][0, 0]["times"][0].flatten()

    
    '''Sleep'''
    dict_={}
    dt = cell_metrics_structure[0,0]['general']['states'][0,0][0]['SleepState'][0].dtype.names
    for dn in dt:
        dict_[dn] =  cell_metrics_structure[0,0]['general']['states'][0,0][0]['SleepState'][0][dn][0][0]
  

    return df_cell, df_behav, spikes, dict_



def load_epochs(project_path: str, mice_name : str):
    _,epochs,_,_ = load_cell_metrics(project_path, mice_name)
    return epochs



def load_SleepState_states(project_path: str, mice_name : str):
    _,_,_,BrainState = load_cell_metrics(project_path, mice_name)
    return BrainState






def load_spikes(
    project_path,
    mice_name,
    putativeCellType=[],  # restrict spikes to putativeCellType
    brainRegion=[],  # restrict spikes to brainRegion
    brain_state=[],  # restrict spikes to brainstate
    support=None, 
):
    

    fs_dat= 30000
    if not isinstance(putativeCellType, list):
        putativeCellType = [putativeCellType]
    if not isinstance(brainRegion, list):
        brainRegion = [brainRegion]

    # load cell metrics and spike data
    cell_metrics,epochs, data, brainstate = load_cell_metrics(project_path, mice_name)

    if cell_metrics is None or data is None:
        return None, None

    # put spike data into array st
    st = np.array(data, dtype=object)

    # restrict cell metrics
    if len(putativeCellType) > 0:
        restrict_idx = []
        for cell_type in putativeCellType:
            restrict_idx.append(
               cell_metrics['CellType'].str.contains(cell_type).values
            )
        restrict_idx = np.any(restrict_idx, axis=0)
        cell_metrics = cell_metrics[restrict_idx]
        st = st[restrict_idx]

    

    if len(brainRegion) > 0:
        restrict_idx = []
        
        for brain_region in brainRegion:
            if len(brain_region)==3:
                    
                restrict_idx.append(
                    cell_metrics['BrainRegion'].str.contains(brain_region).values
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
                        cell_metrics_regions['BrainRegion'].str.contains(b_region).values
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
        result_array[:len(arr), i] = arr[:,0]
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
        brain_states = ['WAKEstate', 'NREMstate', 'REMstate', 'WAKEtheta',
                        'WAKEnontheta', 'WAKEtheta_ThDt', 'REMtheta_ThDt',
                        'QWake_ThDt', 'QWake_noRipples_ThDt', 'NREM_ThDt',
                        'NREM_noRipples_ThDt']
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
