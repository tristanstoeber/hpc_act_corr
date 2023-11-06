import pandas as pd

import dataconvv
import quantities as pq


dataset = "hc-3"
path = "D:/Research/Course/T/data/"
animal_id = "ec012ec"
day = 11
beh = "Mwheel"
session = 187
unit_spiketime = pq.ms
unit_space = pq.mm
load_lfp = False  # Set to True if you want to load LFP data


data = dataconvv.ReadSession(dataset,path,animal_id,day,beh,session,unit_spiketime,unit_space,load_lfp)
# data.df_cells("")
times = data.df_session
clusters = data.df_cells
blk = data.blk
