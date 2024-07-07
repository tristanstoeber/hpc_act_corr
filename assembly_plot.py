
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib import gridspec, rcParams
from matplotlib.colors import Normalize
import seaborn as sns

def plot_assembly_activity(assembly_activities, epoch_names):
    num_epochs = len(epoch_names)
    fig, axs = plt.subplots(1, num_epochs, figsize=(20, 5))
    for i, (epoch_name, activity) in enumerate(assembly_activities.items()):
        plt.style.use('seaborn-v0_8-pastel')
        csfont = {'fontname':'Comic Sans MS'}
        hfont = {'fontname':'Helvetica'}
        time_values = np.arange(activity.shape[1]) / 1000  # Convert time to seconds
        axs[i].plot(time_values, activity.T)
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Assembly Activity')
        axs[i].set_title(f'Activity in {epoch_name}')
        #axs[i].legend(range(len(activity)), loc='upper right')
        #axs[0].format(xticks=20, xtickminor=False) 
        #axs.format(suptitle='ProPlot API', title='Title',
           #xlabel='x axis', ylabel='y axis')
        axs[i].grid(False)
        axs[i].text(-0.1, 1.05, chr(65+i), transform=axs[i].transAxes, size=20, weight='bold')
    plt.tight_layout()
    plt.show()


def plot_combined_assembly_activity(assembly_activities, epoch_names):
    plt.figure(figsize=(12, 6))
    for epoch_name, activity in assembly_activities.items():
        for i in range(activity.shape[0]):
            plt.style.use('seaborn-v0_8-pastel')
            time_values = np.arange(activity.shape[1]) / 1000
            plt.plot(time_values, activity[i], label=f'{epoch_name} - Assembly {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Assembly Activity')
    plt.title('Combined Assembly Activity across Epochs')
    plt.legend(loc='upper right')
    plt.grid(False)
    plt.show()



def plot_assembly_patterns(patterns, epoch_names, ca2_indices, ca3_indices, ca1_indices):
    assembly_counts = {'CA1 only': [], 'CA2 only': [], 'CA3 only': [], 'CA2-CA3': [], 'CA1-CA2': []}
    pattern_types = {epoch_name: {'CA1 only': [], 'CA2 only': [], 'CA3 only': [], 'CA2-CA3': [], 'CA1-CA2': []} for epoch_name in epoch_names}
    all_indeces = ca2_indices + ca3_indices + ca1_indices
    
    for epoch_name, pattern_set in patterns.items():
        plt.style.use('seaborn-v0_8-pastel')
        num_neurons = len(pattern_set[0])
        threshold = abs(1 / np.sqrt(num_neurons))
        
        ca1_count = 0
        ca2_count = 0
        ca3_count = 0
        ca2_ca3_count = 0
        ca1_ca2_count = 0


        a = pd.DataFrame(patterns[epoch_name].T, all_indeces)

        for j, pattern in enumerate(pattern_set):
            above_threshold_ser = a[j].where(a[j] > threshold)
            above_threshold = above_threshold_ser.dropna().index.tolist()
            print(f"Epoch: {epoch_name}, Assembly {j + 1} - Neurons above threshold: {above_threshold}")

            if any(neuron in ca2_indices for neuron in above_threshold):
                if all(neuron in ca2_indices for neuron in above_threshold):
                    ca2_count += 1
                    pattern_types[epoch_name]['CA2 only'].append(j)
                elif any(neuron in ca3_indices for neuron in above_threshold):
                    ca2_ca3_count += 1
                    pattern_types[epoch_name]['CA2-CA3'].append(j)
                elif any(neuron in ca1_indices for neuron in above_threshold):
                    ca1_ca2_count += 1
                    pattern_types[epoch_name]['CA1-CA2'].append(j)

            if any(neuron in ca3_indices for neuron in above_threshold):
                if all(neuron in ca3_indices for neuron in above_threshold):
                    ca3_count += 1
                    pattern_types[epoch_name]['CA3 only'].append(j)

            if any(neuron in ca1_indices for neuron in above_threshold):
                if all(neuron in ca1_indices for neuron in above_threshold):
                    ca1_count += 1
                    pattern_types[epoch_name]['CA1 only'].append(j)

        assembly_counts['CA1 only'].append(ca1_count)
        assembly_counts['CA2 only'].append(ca2_count)
        assembly_counts['CA3 only'].append(ca3_count)
        assembly_counts['CA2-CA3'].append(ca2_ca3_count)
        assembly_counts['CA1-CA2'].append(ca1_ca2_count)

    return assembly_counts, pattern_types


def normalize_patterns(patterns):
    for k in range(patterns.shape[1]):
        pattern = patterns[:, k]
        
        #scale to unit lengtn
        pattern = pattern/np.linalg.norm(pattern)
        if np.max(np.abs(pattern)) != pattern[np.argmax(np.abs(pattern))]:
            pattern = -pattern
        patterns[:,k] = pattern
    return patterns
     


def plot_assembly_counts(assembly_counts, epoch_names):
    labels = epoch_names
    ca1_only = assembly_counts['CA1 only']
    ca2_only = assembly_counts['CA2 only']
    ca3_only = assembly_counts['CA3 only']
    ca2_ca3 = assembly_counts['CA2-CA3']
    ca1_ca2 = assembly_counts['CA1-CA2']

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - 2 * width, ca1_only, width, label='CA1 only')
    rects2 = ax.bar(x - width, ca2_only, width, label='CA2 only')
    rects3 = ax.bar(x, ca3_only, width, label='CA3 only')
    rects4 = ax.bar(x + width, ca2_ca3, width, label='CA2-CA3')
    rects5 = ax.bar(x + 2 * width, ca1_ca2, width, label='CA1-CA2')
   

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Number of Assemblies')
    ax.set_title('Number of Assemblies by Region and Epoch')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.xticks()
    fig.tight_layout()
    plt.show()


def plot_assembly_activity_overtime(assembly_activities, epoch_lengths):
    intervals = []
    start = 0
    for length in epoch_lengths:
        end = start + length
        intervals.append((start, end))
        start = end
    # Merge all activities into a single array
    merged_activities = np.concatenate([v for v in assembly_activities.values()], axis=1)
    # Create a figure and plot
    fig = plt.figure(figsize=(20, 8))
    
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    axs = []
    gs = gridspec.GridSpec(2, 1, height_ratios=[8, 1])
    axs.append(fig.add_subplot(gs[0]))
    axs.append(fig.add_subplot(gs[1]))
    #time_values = np.arange(assembly_activities.shape[1]) / 1000
    #axs[0].plot(merged_activities.T, alpha = 0.3, color = 'black')
    sns.lineplot(merged_activities.T, ax = axs[0], legend = False )
    axs[0].axvspan(intervals[3][0], intervals[3][1], alpha=0.1, color="red")
    axs[0].axvspan(intervals[5][0], intervals[5][1], alpha=0.1, color="brown")
    axs[0].axvspan(intervals[1][0], intervals[1][1], alpha=0.1, color="orange")
    axs[0].axvspan(intervals[8][0], intervals[7][1], alpha=0.1, color="olive")
    axs[0].set_xlim(0,merged_activities.shape[1])
    # Add vertical spans for each epoch
    behavior = ['rest_hab_pre','habituation_arena','rest_hab_post','habituation_cage','rest_pre',
                'exposure_Novel','exposure_Reverse','rest_post2','1novel_exposure','rest_post1']
    colors = list(mcolors.TABLEAU_COLORS.values())
    b = 0.025
    for i, (start, end) in enumerate(intervals):
        axs[1].axvspan(start, end, alpha=0.3, color=colors[i % len(colors)], label=list(assembly_activities.keys())[i])
        axs[1].yaxis.set_major_locator(ticker.NullLocator())
        axs[1].set_xlim(0,intervals[i][1])
        plt.xticks(np.arange(0, intervals[i][1], step=30000), labels=[str(int(i*b//60)) for i in np.arange(0, intervals[i][1], step=30000)])
        axs[1].legend(labels=behavior, loc="lower right", bbox_to_anchor=(0, -2, 1, 0.1), ncols=len(behavior), mode="expand", borderaxespad=0, fontsize=8)
        axs[1].set_xlabel('time (min)')

    #plt.xlabel('Time')
    plt.ylabel('Activity')
    #plt.title('Activity of Cell Assemblies Over Time')
    #plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1)
    #plt.tight_layout()
    plt.show()