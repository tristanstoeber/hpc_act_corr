import numpy as np
import pandas as pd
import scipy.io
from scipy import stats
import statistics
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib import gridspec, rcParams

def correlate(spikes_bin_1, spikes_bin_2):
    correlation = np.correlate(spikes_bin_1, spikes_bin_2).tolist()[0]
    return correlation


def cross_correlate(spikes_bin_1, spikes_bin_2, bin):
    lags = range(int(2/(bin*2)))
    cross_correlation = []
    times = []
    
    for lag in lags:
        if lag == 0:
            correlation = correlate(spikes_bin_1, spikes_bin_2)
            cross_correlation.append(correlation)
            times.append(0)
        else:
            correlation = correlate(spikes_bin_1[lag:], spikes_bin_2[:len(spikes_bin_2)-lag])
            cross_correlation.insert(0, correlation)
            times.insert(0, -lag*bin)
            correlation = correlate(spikes_bin_1[:len(spikes_bin_1)-lag], spikes_bin_2[lag:])
            cross_correlation.append(correlation)
            times.append(lag*bin)
    
    if statistics.mean(cross_correlation) == 0:
        percentage = 100
    else:
        percentage = ((cross_correlation[times.index(0)] - statistics.mean(cross_correlation)) / statistics.mean(cross_correlation)) * 100
    return cross_correlation, times, percentage


def get_continuous(spikes_bin):
    behaviors = []
    behavior_intervals = []
    start = 0
    con_spikes_bin = []
    for behavior in list(spikes_bin.keys()):
        behaviors.append(behavior)
        end = start + len(spikes_bin[behavior].to_numpy())
        interval = [start, end]
        behavior_intervals.append(interval)
        start = end
        con_spikes_bin.append(spikes_bin[behavior])
    
    con_spikes_bin_times = [round(t, 3) for t in pd.concat(con_spikes_bin).index.tolist()]
    con_spikes_bin = pd.concat(con_spikes_bin).to_numpy().transpose()
    return behaviors, behavior_intervals, con_spikes_bin, con_spikes_bin_times


def get_epochs_1(times, spikes_bin, bin, epoch, overlap):
    epochs = []
    start = 0
    while start < times[-1]:
        end = start + epoch
        epochs.append([start, end])
        start = end - overlap
    ep_spikes_bin = []
    for spike_bin in spikes_bin:
        ep_spike_bin = []
        for time_in in epochs:
            time_in_index = []
            for i, t in enumerate(np.array(times)):
                if time_in[0] - bin/2 <= t <= time_in[1] + bin/2:
                    time_in_index.append(i)
            ep_spike_bin.append(np.array(spike_bin)[time_in_index])
        ep_spikes_bin.append(ep_spike_bin)
    epoch_intervals = [[j/bin for j in i] for i in epochs]   
    return epoch_intervals, ep_spikes_bin

def get_epochs_2(times, spikes_bin, bin, epoch, overlap, behavior_intervals, first, second):
    epochs = []
    start = behavior_intervals[first][1]*bin - 2400
    for n in range(4):
        end = start + epoch
        epochs.append([start, end])
        start = end - overlap
    start = behavior_intervals[second][0]*bin
    for n in range(4):
        end = start + epoch
        epochs.append([start, end])
        start = end - overlap
    
    ep_spikes_bin = []
    for spike_bin in spikes_bin:
        ep_spike_bin = []
        for time_in in epochs:
            time_in_index = []
            for i, t in enumerate(np.array(times)):
                if time_in[0] - bin/2 <= t <= time_in[1] + bin/2:
                    time_in_index.append(i)
            ep_spike_bin.append(np.array(spike_bin)[time_in_index])
        ep_spikes_bin.append(ep_spike_bin)
    return ep_spikes_bin

def get_ep_correlation(ep_spikes_bin_1, ep_spikes_bin_2, bin):
    cross_correlations = []
    ep_percentages = []
    for ep_spike_bin_1 in ep_spikes_bin_1:
        for ep_spike_bin_2 in ep_spikes_bin_2:
            ep_percentage = []
            for m in range(len(ep_spike_bin_2)):
                cross_correlation, times, percentage = cross_correlate(ep_spike_bin_1[m], ep_spike_bin_2[m], bin)
                cross_correlations.append(cross_correlation)
                ep_percentage.append(percentage)
            ep_percentages.append(ep_percentage)
    return ep_percentages, cross_correlations, times
    

def get_ep_frequency(ep_spikes_bin, bin):
    ep_frequencies = []
    for ep_spike_bin in ep_spikes_bin:
        ep_frequency = []
        for spike_bin in ep_spike_bin:
            frequency = np.sum(spike_bin) / (spike_bin.shape[0] * bin)
            ep_frequency.append(frequency)
        ep_frequencies.append(ep_frequency)
    return ep_frequencies


def get_SWR(project_path, mice_name):
    mat_data_SWR = scipy.io.loadmat(project_path+ '/'+ mice_name +'/'+'SWR.mat')
    SWR_0 = mat_data_SWR["SWR"][0][0][0]
    SWR_1 = mat_data_SWR["SWR"][0][0][1]
    return SWR_0, SWR_1


def plot_frequency_correlation(behavior_intervals, con_spikes_bin_ca2, con_spikes_bin_ca3, epoch_intervals, ep_correlations_ca2_ca3, bin, epoch, behaviors, SWR_peaktimes, SleepState, name):
    
    mean_ep_correlations_ca2_ca3 = np.mean(np.array(ep_correlations_ca2_ca3).transpose()[:-1], axis=1)
    std_ep_correlations_ca2_ca3 = np.std(np.array(ep_correlations_ca2_ca3).transpose()[:-1], axis=1)
    
    mean_epoch_intervals = np.mean(np.array(epoch_intervals[:-1]), axis=1)
    
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    rcParams.update({'font.size': 12})

    fig = plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(6, 1, height_ratios=[8, 8, 4, 2, 2, 1])
    axs = []
    axs.append(fig.add_subplot(gs[0]))
    axs.append(fig.add_subplot(gs[1]))
    axs.append(fig.add_subplot(gs[2]))
    axs.append(fig.add_subplot(gs[3]))
    axs.append(fig.add_subplot(gs[4]))
    axs.append(fig.add_subplot(gs[5]))


    axs[0].imshow(con_spikes_bin_ca2,
               aspect='auto',
               vmax=0.09,
               vmin=-0.06)
    axs[0].xaxis.set_major_locator(ticker.NullLocator())
    axs[0].set_ylabel('CA2 neural activity', fontsize=14)
    axs[1].imshow(con_spikes_bin_ca3,
               aspect='auto',
               vmax=0.09,
               vmin=-0.06)
    axs[1].set_ylabel('CA3 neural activity', fontsize=14)
    axs[1].xaxis.set_major_locator(ticker.NullLocator())
    
    axs[2].scatter(mean_epoch_intervals, mean_ep_correlations_ca2_ca3, alpha=0.5)
    axs[2].errorbar(mean_epoch_intervals, mean_ep_correlations_ca2_ca3, yerr=std_ep_correlations_ca2_ca3, fmt="o", alpha=0.5)
    rect_1 = patches.Rectangle((mean_epoch_intervals[1] - epoch/(bin*2),0), epoch/(bin), max(np.mean(ep_correlations_ca2_ca3, 0)), linewidth=0.5, edgecolor='r', facecolor='none')
    rect_2 = patches.Rectangle((mean_epoch_intervals[2] - epoch/(bin*2),0), epoch/(bin), max(np.mean(ep_correlations_ca2_ca3, 0)), linewidth=0.5, edgecolor='black', facecolor='none')
    axs[2].add_patch(rect_1)
    axs[2].add_patch(rect_2)
    for n in [4,7]:
        axs[2].axvspan(behavior_intervals[n][0], behavior_intervals[n][1], alpha=0.2, color=list(mcolors.TABLEAU_COLORS.values())[n])
    axs[2].set_ylabel("CCG (%)", fontsize=15)
    axs[2].set_xlim(0,behavior_intervals[-1][1])
    axs[2].xaxis.set_major_locator(ticker.NullLocator())

    axs[3].scatter(SWR_peaktimes, np.ones_like(SWR_peaktimes), s=300, marker=2, alpha=0.1, color="b")
    axs[3].set_ylim(0.95, 1.2)
    axs[3].set_xlim(0, behavior_intervals[-1][1]*bin)
    axs[3].xaxis.set_major_locator(ticker.NullLocator())
    axs[3].yaxis.set_major_locator(ticker.NullLocator())
    axs[3].set_ylabel('SWR', fontsize=15)
    for n in [4,7]:
        axs[3].axvspan(behavior_intervals[n][0]*bin, behavior_intervals[n][1]*bin, alpha=0.2, color=list(mcolors.TABLEAU_COLORS.values())[n])
        
    for WAKE in SleepState[0]:
        axs[4].axvspan(WAKE[0], WAKE[1], 0.7, 1, color="violet")
    for REM in SleepState[1]:
        axs[4].axvspan(REM[0], REM[1], 0.35, 0.65, color="cornflowerblue")
    for NREM in SleepState[2]:
        axs[4].axvspan(NREM[0], NREM[1], 0, 0.3, color="slategrey")
    
    axs[4].set_xlim(0, SWR_peaktimes[-1])
    axs[4].xaxis.set_major_locator(ticker.NullLocator())
    axs[4].set_yticks([0.15, 0.5, 0.85], labels=["NREM", "REM", "WAKE"], fontsize=12)

    for n in range(len(behaviors)):
        axs[5].axvspan(behavior_intervals[n][0], behavior_intervals[n][1], alpha=0.5, color=list(mcolors.TABLEAU_COLORS.values())[n])  
    axs[5].yaxis.set_major_locator(ticker.NullLocator())
    axs[5].set_xlim(0, behavior_intervals[-1][1])

    axs[5].legend(labels=['rest_hab_pre','habituation_arena','rest_hab_post','habituation_cage','rest_pre','2novel_exposure','exposure_reversed','rest_post2','1novel_exposure','rest_post1'][:len(behaviors)], loc="lower right", bbox_to_anchor=(0, -4, 1, 0.1), ncols=len(behaviors), mode="expand", borderaxespad=0, fontsize=10)
    axs[5].set_xlabel('time (min)', fontsize=15)
    axs[5].set_xticks(np.arange(0, behavior_intervals[-1][1], step=600//bin), labels=[str(int(i*bin//60)) for i in np.arange(0, behavior_intervals[-1][1], step=600//bin)])
    plt.title(f"Neural activity, correlation strength and SWR during different behavioral paradigms and sleep states in {name}", pad=570, fontsize=20)
    
    plt.show()
    plt.rcdefaults() 




def get_none_with_SWR(times, spikes_bin, SWR_timestamps, bin):    
    time_ex_index = []
    for time_ex in SWR_timestamps:
        for i, t in enumerate(np.array(times)):
            if time_ex[0] - bin/2 <= t <= time_ex[1] + bin/2:
                time_ex_index.append(i)
    time_ex_index = np.sort(list(set(time_ex_index))).tolist()
    none_times = np.delete(times, time_ex_index, 0).tolist()
    with_times = np.round(np.array(times)[time_ex_index], 3).tolist()
    none_spikes_bin = np.delete(spikes_bin, time_ex_index, 1)
    with_spikes_bin = np.array(spikes_bin)[:,time_ex_index]

    return none_times, with_times, none_spikes_bin, with_spikes_bin


def plot_correlation_1(social, object):
    times = [-40,-30,-20,-10,10,20,30,40]
    fig, axs = plt.subplots(figsize=(4,4))
    axs.errorbar(times, np.mean(social, 0), yerr=stats.sem(social, 0), marker='o', color='red')
    axs.errorbar(times, np.mean(object, 0), yerr=stats.sem(object, 0) , marker='s', color='black')
    axs.legend(labels=["social interaction","object interaction"], loc="upper left")
    axs.set_xlabel("Time (min)")
    axs.set_ylabel("CCG outside SWR (%)")
    axs.set_xlim(-60, 60)
    axs.set_ylim(0, 300)
    plt.title("Correlation strength following social interaction with unfamiliar mice or interaction with a novel object outside SWR episodes", pad=20)

    plt.show()

def plot_correlation_2(social, object):
    times = [-40,-30,-20,-10,10,20,30,40]
    fig, axs = plt.subplots(figsize=(4,4))
    axs.errorbar(times, np.mean(social, 0), yerr=stats.sem(social, 0), marker='o', color='red')
    axs.errorbar(times, np.mean(object, 0), yerr=stats.sem(object, 0) , marker='s', color='black')
    axs.legend(labels=["social interaction","object interaction"], loc="upper left")
    axs.set_xlabel("Time (min)")
    axs.set_ylabel("CCG during SWR (%)")
    axs.set_xlim(-60, 60)
    axs.set_ylim(0, 300)
    plt.title("Correlation strength following social interaction with unfamiliar mice or interaction with a novel object during SWR episodes", pad=20)

    plt.show()



def plot_frequency(ca2, ca3):
    times = [-40,-30,-20,-10,10,20,30,40]
    fig, axs = plt.subplots(figsize=(4,4))
    axs.errorbar(times, np.mean(ca2, 0), yerr=stats.sem(ca2, 0), marker='o', color='red')
    axs.errorbar(times, np.mean(ca3, 0), yerr=stats.sem(ca3, 0), marker='s', color='black')
    axs.legend(labels=["CA2","CA3"], loc="upper left")
    axs.set_xlabel("Time (min)")
    axs.set_ylabel("Firing frequency (Hz)")
    axs.set_xlim(-60, 60)
    axs.set_ylim(-1, 4)
    plt.title("Average firing rates of CA3 and CA2 cells before and after social interaction with unfamiliar mice", pad=20)

    plt.show()

