import scipy.io
import pynapple as nap
import pandas as pd
import numpy as np
import os


##For Loading data
##Written by Fatemeh Jamshidian


def process_epochs_and_neurons(project_path,mice_name):
        
    mat_data_cell = scipy.io.loadmat(project_path+ '/'+ mice_name+'/'+'cell_id_region.mat')
    mat_data_behav = scipy.io.loadmat(project_path+ '/'+ mice_name+'/'+'behavior_times.mat')
    mat_data_spk = scipy.io.loadmat(project_path+ '/'+ mice_name+'/'+mice_name.split('_')[1]+'.spikes.cellinfo.mat')
      
    
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
    
    df_behav['Cage_or_Maze'] = df_behav['Cage_or_Maze'].map(behav_map)
    
    '''spkikes'''
    
    spk_times= mat_data_spk['spikes'][0,0].flatten()
    # spikes_structure = spk_times[0]
    # spikes_dict = dict(zip(spikes_structure.dtype.names, spikes_structure))
    # spikes = spikes_dict['times'].flatten()
    
    #df_spk = pd.DataFrame(spk_times[spk_times.dtype.names[2]][0][0].flatten())
    
    list_of_dfs = [pd.DataFrame(time) for i, time in enumerate(spk_times[spk_times.dtype.names[2]][0][0].flatten())]
    df_spike_times = pd.concat(list_of_dfs, axis=1)
    
    
    
    neurons_data = nap.TsGroup({df_cell['CluID'][i]: nap.Ts(t=df_spike_times.values[:, i], time_units="s") for i in range(len(df_cell))},
                            BrainRegion= df_cell.values.T[1], 
                            CluID = df_cell.values.T[3],
                            CellType = df_cell.values.T[0])
    
    all_bincounted= {}
    all_bincounted_sorted = {}
    all_bincounted_p = {}
    Ca2_p_bincounted = {}
    Ca3_p_bincounted = {}
    Ca1_p_bincounted = {}
    Ca2_inter_bincounted = {}
    Ca3_inter_bincounted = {}
    Ca1_inter_bincounted = {}
    total_epoch= {}

    for i in range(len(df_behav)):
        row = df_behav.iloc[i]
        if i == 0:
            total_epoch['Start_Time']= row['Start_Time']
        elif i == (len(df_behav)-1):
            total_epoch['End_Time']= row['End_Time']

        
        behavioral_paradigm = row['Cage_or_Maze']
        epoch = nap.IntervalSet(start=row['Start_Time'], end=row['End_Time'], time_units='s')
        ts_epoch = neurons_data.restrict(epoch)
        bincount = neurons_data.count(0.025, epoch).as_dataframe()
        if "CA3_id" not in locals():
            #Pyramidal
            try:
                CA2_spk_p = ts_epoch.getby_category("BrainRegion")["CA2"].getby_category("CellType")["pyr"]
                CA2_id_p = list(CA2_spk_p.index)
    
    
            except:
                CA2_id_p = []
                
            try:
                CA3_spk_p = ts_epoch.getby_category("BrainRegion")["CA3"].getby_category("CellType")["pyr"]
                CA3_id_p = list(CA3_spk_p.index)
            
            except:
                CA3_id_p  = []
                
            try:
                CA1_spk_p = ts_epoch.getby_category("BrainRegion")["CA1"].getby_category("CellType")["pyr"]
                CA1_id_p = list(CA1_spk_p.index)
            
            except:
                CA1_id_p  = []
                
            #Interneuron
            try:
                CA2_spk_ni = ts_epoch.getby_category("BrainRegion")["CA2"].getby_category("CellType")["int"]
                CA2_id_inter = list(CA2_spk_ni.index)
    
    
            except:
                CA2_id_inter = []
                
            try:
                CA3_spk_ni = ts_epoch.getby_category("BrainRegion")["CA3"].getby_category("CellType")["int"]
                CA3_id_inter = list(CA3_spk_ni.index)
            
            except:
                CA3_id_inter  = []
                
                
            try:
                CA1_spk_inter = ts_epoch.getby_category("BrainRegion")["CA1"].getby_category("CellType")["int"]
                CA1_id_inter = list(CA1_spk_inter.index)
            
            except:
                CA1_id_inter  = []
         
            
            
            sorted_columns =   CA2_id_p  + CA3_id_p + CA1_id_p+ CA2_id_inter + CA3_id_inter + CA1_id_inter
            sorted_p = CA2_id_p  + CA3_id_p + CA1_id_p
            all_bincounted[mice_name+"_" + behavioral_paradigm] = bincount
            all_bincounted_p[mice_name+"_" + behavioral_paradigm] = bincount.reindex(columns = sorted_p)[sorted_p]
            
            Ca2_p_bincounted[mice_name+"_" + behavioral_paradigm ] = bincount.reindex(columns=CA2_id_p)[CA2_id_p]
            Ca3_p_bincounted[mice_name+"_" + behavioral_paradigm ] = bincount.reindex(columns=CA3_id_p)[CA3_id_p]
            Ca1_p_bincounted[mice_name+"_" + behavioral_paradigm ] = bincount.reindex(columns=CA1_id_p)[CA1_id_p]

            
            Ca2_inter_bincounted[mice_name+"_" + behavioral_paradigm ] = bincount.reindex(columns=CA2_id_inter)[CA2_id_inter]
            Ca3_inter_bincounted[mice_name+"_" + behavioral_paradigm ] = bincount.reindex(columns=CA3_id_inter)[CA3_id_inter]
            Ca1_inter_bincounted[mice_name+"_" + behavioral_paradigm ] = bincount.reindex(columns=CA1_id_inter)[CA1_id_inter]
            
            all_bincounted_sorted[mice_name+"_" + behavioral_paradigm ] = bincount.reindex(columns=sorted_columns)[sorted_columns]
    
    
    epoch_t = nap.IntervalSet(start=total_epoch['Start_Time'], end=total_epoch['End_Time'], time_units='s')
    ts_epoch_t = neurons_data.restrict(epoch_t)
    bincount_t = neurons_data.count(0.025, epoch_t).as_dataframe()
    total_bincounted = bincount_t.reindex(columns = sorted_p)[sorted_p]
    
    
    return  all_bincounted_p , Ca2_p_bincounted, Ca3_p_bincounted, Ca1_p_bincounted, CA2_id_p, CA3_id_p, CA1_id_p
