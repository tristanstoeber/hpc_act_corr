import numpy as np
import sqlite3
import pandas as pd
import re
import tarfile as tf
import neo
import quantities as pq
import xml.etree.ElementTree as et
import helpfuncs as hf
import pdb


def regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None


class ReadSession:
    def __init__(self,
                 dataset,
                 path,
                 animal_id,
                 day,
                 beh,
                 session,
                 unit_spiketime,
                 unit_space,
                 unit_lfp=pq.V,
                 load_lfp=False):

        meta_path = path + dataset + '/docs/' + dataset.replace('-', '') +\
                    '-metadata-tables/' + dataset.replace('-', '') +\
                    '-tables.db'

        # Open database
        con_sessions = sqlite3.connect(meta_path)
        con_sessions.create_function("REGEXP", 2, regexp)

        topdir = animal_id + '.' + str(day)
        subdir = animal_id + '.' + str(session)
        
        df_session = pd.read_sql_query(
            'SELECT ' +
            'topdir, ' +
            'session, ' +
            'behavior, ' +
            'familiarity, ' +
            'duration ' +
            'from session where behavior=\'' +
            beh +
            '\' AND session=\'' +
            subdir +
            '\' AND topdir=\'' +
            topdir +
            '\'', con_sessions)

        df_cells = pd.read_sql_query(
            'SELECT ' +
            'id, ' +
            'topdir, ' +
            'animal, ' +
            'ele, ' +
            'clu, ' +
            'region, ' +
            'nexciting, ' +
            'ninhibiting, ' +
            'exciting, ' +
            'inhibiting, ' +
            'excited, ' +
            'inhibited, ' +
            'fireRate, ' +
            'totalFireRate, ' +
            'cellType ' +
            'From cell where topdir REGEXP \'' +
            topdir + '\'',
            con_sessions)
        
        df_epos = pd.read_sql_query(
            'SELECT ' +
            'topdir, ' +
            'animal, ' +
            'e1, ' +
            'e2, ' +
            'e3, ' +
            'e4, ' +
            'e5, ' +
            'e6, ' +
            'e7, ' +
            'e8, ' +
            'e9, ' +
            'e10, ' +
            'e11, ' +
            'e12, ' +
            'e13, ' +
            'e14, ' +
            'e15, ' +
            'e16 ' +
            'From epos where topdir REGEXP \'' +
            topdir + '\'',
            con_sessions)
        
        electrode_ids = np.unique(df_cells['ele'])
        path_to_session = path + dataset + '/' + \
            topdir + '/' +\
            subdir + '.tar.gz'

        # extract variables from data
        clusters = {}
        times = {}
        print('Get position and spikes')
        with tf.open(path_to_session) as tf_session:
            # get sampling rate of spike timestamps

            xml_f = tf_session.extractfile(
                topdir + '/' +
                subdir + '/' +
                subdir + '.xml')
            e = et.parse(xml_f).getroot()
            sampling_rate_spike_time = float(
                e.findall("./acquisitionSystem/samplingRate")[0].text)

            # get animal position
            positions_file = tf_session.extractfile(
                topdir + '/' +
                subdir + '/' +
                subdir + '.whl')
            positions_file_lines = [np.array(line.split(), dtype=np.float)
                                    for line in positions_file.readlines()]
            positions = np.stack(positions_file_lines)
            for ele_i in electrode_ids:
                clusters_f = tf_session.extractfile(
                    topdir + '/' +
                    subdir + '/' +
                    subdir + '.clu.' + str(ele_i))
                # read cluster file
                clusters_i = np.array([
                    int(clu_id) for clu_id in clusters_f.readlines()])
                # first line contains number of clusters in file, skip it
                clusters_i = clusters_i[1:]
                times_f = tf_session.extractfile(
                    topdir + '/' +
                    subdir + '/' +
                    subdir + '.res.' + str(ele_i))
                # get times of spikes
                times_i = np.array([
                    float(time_j) for time_j
                    in times_f.readlines()])*unit_spiketime
                # divide by sampling rate
                times_i /= sampling_rate_spike_time
                
                # from documentation:
                # cluster 0 corresponds to mechanical noise (the wave shapes
                # do not look like neuron's spike). Cluster 1 corresponds to
                # small, unsortable spikes. These two clusters (0 and 1) should
                # not be used for analysis of neural data since they do not
                # correspond to successfully sorted spikes.

                # remove clusters == 0 and == 1
                pos_cluster_not_0_or_1 = np.where(clusters_i >= 2)[0]
                clusters_i = clusters_i[pos_cluster_not_0_or_1]
                times_i = times_i[pos_cluster_not_0_or_1]
                clusters[ele_i] = clusters_i
                times[ele_i] = times_i

            if load_lfp:
                xml = hf.etree_to_dict(e)['parameters']
                dtype_str = xml['acquisitionSystem']['nBits']
                if dtype_str == '16':
                    lfp_dtype = np.int16
                lfp_n_channels = int(xml['acquisitionSystem']['nChannels'])
                lfp_samplingrate = int(xml['fieldPotentials']['lfpSamplingRate']) * pq.Hz
                lfp_amplification = 1./np.float(xml['acquisitionSystem']['amplification'])
                eeg_f = tf_session.extractfile(
                    topdir + '/' +
                    subdir + '/' +
                    subdir + '.eeg')
                eeg_f_content = eeg_f.read()
                lfp_raw = np.fromstring(eeg_f_content, dtype=np.int16)
                #                lfp_raw = np.fromfile(eeg_f, dtype=np.int16)
                lfp = np.zeros((int(len(lfp_raw)/lfp_n_channels), lfp_n_channels))
                for i in range(lfp_n_channels):
                    lfp[:, i] = lfp_raw[i::lfp_n_channels]
                lfp *= lfp_amplification
                lfp *= unit_lfp
                lfp_n_electrodes = len(xml['anatomicalDescription']['channelGroups']['group'])-1
                lfp_channel_groups = []
                for i in range(lfp_n_electrodes):
                    channel_group_i = []
                    for row in xml['anatomicalDescription']['channelGroups']['group'][i]['channel']:
                        channel_group_i.append(int(row['#text']))
                    lfp_channel_groups.append(channel_group_i)
                        # each entry in channel groups, corresponds to an electrode number
                
                    
        positions = positions * unit_space
        # create neo structure to hold represent data.
        blk = neo.Block(animal_id=animal_id,
                        dataset=dataset,
                        path=path,
                        day=day,
                        beh=beh,
                        topdir=topdir,
                        subdir=subdir,
                        session=session)
        seg = neo.Segment(name='asdf')
        blk.segments.append(seg)
        chx_units = neo.ChannelIndex(index=0,
                                     name='units')
        blk.channel_indexes.append(chx_units)
        chx_pos = neo.ChannelIndex(index=1,
                                   name='position')
        blk.channel_indexes.append(chx_pos)
        sampling_rate_pos = 39.0625 * pq.Hz
        pos_led0_x = neo.AnalogSignal(positions[:, 0].rescale('m'),
                                      sampling_rate=sampling_rate_pos,
                                      name='led0_x',
                                      led_pos='anterior')
        chx_pos.analogsignals.append(pos_led0_x)
        pos_led0_y = neo.AnalogSignal(positions[:, 1].rescale('m'),
                                      sampling_rate=sampling_rate_pos,
                                      name='led0_y',
                                      led_pos='anterior')
        chx_pos.analogsignals.append(pos_led0_y)
        pos_led1_x = neo.AnalogSignal(positions[:, 2].rescale('m'),
                                      sampling_rate=sampling_rate_pos,
                                      name='led1_x',
                                      led_pos='posterior')
        chx_pos.analogsignals.append(pos_led1_x)
        pos_led1_y = neo.AnalogSignal(positions[:, 3].rescale('m'),
                                      sampling_rate=sampling_rate_pos,
                                      name='led1_y',
                                      led_pos='posterior')
        chx_pos.analogsignals.append(pos_led1_y)
        
        for _, row_i in df_cells.iterrows():
            # create unit
            unit_i = neo.Unit(name=row_i['id'],
                              electrode=row_i['ele'],
                              cluster=row_i['clu'],
                              region=row_i['region'],
                              exciting=row_i['exciting'],
                              inhibiting=row_i['inhibiting'],
                              cellType=row_i['cellType'])
            # get spike times
            ele = row_i['ele']
            clu = row_i['clu']
            clusters_i = clusters[ele]
            times_i = times[ele]
            pos_spks = np.where(clusters_i == clu)[0]
            if len(pos_spks) > 0:
                ts_spks = times_i[pos_spks]
                t_stop = df_session['duration'].item()*pq.s
                t_stop = t_stop.rescale(unit_spiketime)
                train_i = neo.SpikeTrain(times=ts_spks,
                                         units=unit_spiketime,
                                         t_start=0*unit_spiketime,
                                         t_stop=t_stop)
                train_i.unit = unit_i
                unit_i.spiketrains.append(train_i)
            chx_units.units.append(unit_i)

        if load_lfp:
            print('Get LFP')
            for i in range(lfp_n_electrodes):
                name = 'e'+str(i+1)
                # get electrode position
                epos_i = df_epos[name].loc[0]
                last_idx = int(blk.channel_indexes[-1].index)
                chx_lfp_i = neo.ChannelIndex(index=last_idx+1+i,
                                             name='lfp_' + name,
                                             channel_indexes=lfp_channel_groups[i],
                                             coordinates=epos_i,
                                             electrode=i+1)
                analog_i = []
                for j in lfp_channel_groups[i]:
                    analog_i.append(lfp[:, j])
                analog_i = np.vstack(analog_i).T
                sig_i = neo.AnalogSignal(analog_i, units=unit_lfp, sampling_rate=lfp_samplingrate)
                chx_lfp_i.analogsignals.append(sig_i)
                blk.channel_indexes.append(chx_lfp_i)
            
        self.df_session = df_session
        self.df_cells = df_cells
        self.dataset = dataset
        self.path = path
        self.animal_id = animal_id
        self.day = day
        self.beh = beh
        self.session = session
        self.unit_spiketime = unit_spiketime
        self.unit_space = unit_space
        self.blk = blk
    
