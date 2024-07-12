
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib import gridspec, rcParams
from matplotlib.colors import Normalize
import warnings
import seaborn as sns
from scipy import stats

##Functions for Plotting assembly, Counting the number of Joint Assemblies###
##Plot assembly activity over all epochs##
##Counting number of cell assemblies##
##ICA weights for each neurons in each cell assembly##
##Written by Armin Toghi and Fatemeh Jamshidian



def plot_assembly_activity(assembly_activities, epoch_names): #Plot assembly activity for a given epoch
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


def normalize_patterns(patterns): #change negative ICA weights to positive (Van de ven et al., 2016)
    for k in range(patterns.shape[1]):
        pattern = patterns[:, k]
        
        #scale to unit lengtn
        pattern = pattern/np.linalg.norm(pattern)
        if np.max(np.abs(pattern)) != pattern[np.argmax(np.abs(pattern))]:
            pattern = -pattern
        patterns[:,k] = pattern
    return patterns
     

def plot_assembly_patterns(patterns, epoch_names, ca2_indices, ca3_indices, ca1_indices): #plot ICA weight: the contribution of each neuron in assembly
    assembly_counts = {'CA1 only': [], 'CA2 only': [], 'CA3 only': [], 'CA2-CA3': [], 'CA1-CA2': []}
    pattern_types = {epoch_name: {'CA1 only': [], 'CA2 only': [], 'CA3 only': [], 'CA2-CA3': [], 'CA1-CA2': []} for epoch_name in epoch_names}
    

    all_indeces = ca2_indices + ca3_indices + ca1_indices
    
    for epoch_name, pattern_set in patterns.items():
        plt.style.use('seaborn-v0_8-pastel')
        num_neurons = len(pattern_set[0])
        
        
        threshold = abs(1 / np.sqrt(num_neurons))
        num_assemblies = len(pattern_set)
        num_cols = num_assemblies
        num_rows = 1
        ca1_count = 0
        ca2_count = 0
        ca3_count = 0
        ca2_ca3_count = 0
        ca1_ca2_count = 0
        
        
        # fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 5))
        # if num_assemblies == 1:
        #     axs = [axs]  # Make axs iterable if it's a single subplot
        

        # for j, pattern in enumerate(pattern_set):
        #     axs[j].stem(pattern, linefmt='-', markerfmt='o', basefmt=' ', label=f'Assembly {j + 1}')
        #     axs[j].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        #     axs[j].set_xlabel('Neuron')
        #     axs[j].set_ylabel('Activation')
        #     axs[j].set_title(f'Pattern {j + 1}')
        #     axs[j].legend()
        #     axs[j].grid(False)
        #     axs[j].text(-0.1, 1.05, chr(65+j), transform=axs[j].transAxes, size=20, weight='bold')
            
            
        # plt.suptitle(f'Patterns in {epoch_name}')
        # plt.tight_layout()
        # plt.show()

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


def plot_assembly_counts(assembly_counts, epoch_names): #Count the number of CA1,CA2,CA3, and Joint Cell Assemblies
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


def plot_assembly_activity_overtime(assembly_activities, epoch_lengths): #Plot the activity of cell assmblies over all epochs
    intervals = []
    start = 0
    for length in epoch_lengths:
        end = start + length
        intervals.append((start, end))
        start = end
    # Merge all activities into a single array
    valid_activities = [v for v in assembly_activities.values()]
    
    if len(valid_activities) > 1:
        merged_activities = np.concatenate(valid_activities, axis=1)
    else:
        merged_activities = valid_activities[:]
        
    #merged_activities = np.concatenate([v for v in assembly_activities.values()], axis=1)
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


def count_cells_per_pattern_type(patterns, pattern_types, ca2_indices, ca3_indices,ca1_indices, threshold):
    cell_counts = {epoch_name: {'CA2': {}, 'CA3': {}} for epoch_name in pattern_types.keys()}

    all_indices = ca2_indices + ca3_indices + ca1_indices

    for epoch_name, pattern_set in patterns.items():
        a = pd.DataFrame(patterns[epoch_name], columns=all_indices)
        
        for pattern_type, assembly_indices in pattern_types[epoch_name].items():
            ca2_counts = {}
            ca3_counts = {}
            
            for j in assembly_indices:
                above_threshold_ser = a.iloc[j].where(a.iloc[j] > threshold)
                above_threshold = above_threshold_ser.dropna().index.tolist()
                
                ca2_count = sum(1 for neuron in above_threshold if neuron in ca2_indices)
                ca3_count = sum(1 for neuron in above_threshold if neuron in ca3_indices)
                
                ca2_counts[str(j)] = ca2_count
                ca3_counts[str(j)] = ca3_count
            
            cell_counts[epoch_name]['CA2'][pattern_type]= ca2_counts
            cell_counts[epoch_name]['CA3'][pattern_type] = ca3_counts
    
    return cell_counts


def plot_multiple_epochs(cell_counts, epoch_names):
    # Prepare data for plotting
    data_ca2_only = []
    data_ca3_only = []
    data_ca2_ca3_ca2 = []
    data_ca2_ca3_ca3 = []
    
    for epoch_name in epoch_names:
        epoch_data = cell_counts.get(epoch_name, {'CA2': {}, 'CA3': {}})
        
        # CA2 only
        ca2_only_counts = epoch_data['CA2'].get('CA2 only', {})
        for idx, count in ca2_only_counts.items():
            data_ca2_only.append((epoch_name, idx, count))

        # CA3 only
        ca3_only_counts = epoch_data['CA3'].get('CA3 only', {})
        for idx, count in ca3_only_counts.items():
            data_ca3_only.append((epoch_name, idx, count))

        # CA2-CA3 for CA2 cells
        ca2_ca3_counts = epoch_data['CA2'].get('CA2-CA3', {})
        for idx, count in ca2_ca3_counts.items():
            data_ca2_ca3_ca2.append((epoch_name, idx, count))

        # CA2-CA3 for CA3 cells
        ca3_ca3_counts = epoch_data['CA3'].get('CA2-CA3', {})
        for idx, count in ca3_ca3_counts.items():
            data_ca2_ca3_ca3.append((epoch_name, idx, count))

    # Convert to DataFrames
    df_ca2_only = pd.DataFrame(data_ca2_only, columns=['Epoch', 'Assembly Index', 'Cell Count'])
    df_ca3_only = pd.DataFrame(data_ca3_only, columns=['Epoch', 'Assembly Index', 'Cell Count'])
    df_ca2_ca3_ca2 = pd.DataFrame(data_ca2_ca3_ca2, columns=['Epoch', 'Assembly Index', 'Cell Count'])
    df_ca2_ca3_ca3 = pd.DataFrame(data_ca2_ca3_ca3, columns=['Epoch', 'Assembly Index', 'Cell Count'])

    # Plot the data
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot CA2 only counts
    if not df_ca2_only.empty:
        sns.violinplot(x='Epoch', y='Cell Count', data=df_ca2_only, ax=axs[0, 0], inner=None, color='yellow')
        sns.swarmplot(x='Epoch', y='Cell Count', data=df_ca2_only, ax=axs[0, 0], color='black', size=3)
        axs[0, 0].set_title('CA2 only assemblies - CA2 cells count')

    # Plot CA3 only counts
    if not df_ca3_only.empty:
        sns.violinplot(x='Epoch', y='Cell Count', data=df_ca3_only, ax=axs[0, 1], inner=None, color='lightcoral')
        sns.swarmplot(x='Epoch', y='Cell Count', data=df_ca3_only, ax=axs[0, 1], color='black', size=3)
        axs[0, 1].set_title('CA3 only assemblies - CA3 cells count')

    # Plot CA2-CA3 CA2 counts
    if not df_ca2_ca3_ca2.empty:
        sns.violinplot(x='Epoch', y='Cell Count', data=df_ca2_ca3_ca2, ax=axs[1, 0], inner=None, color='yellow')
        sns.swarmplot(x='Epoch', y='Cell Count', data=df_ca2_ca3_ca2, ax=axs[1, 0], color='black', size=3)
        axs[1, 0].set_title('CA2-CA3 assemblies - CA2 cells count')

    # Plot CA2-CA3 CA3 counts
    if not df_ca2_ca3_ca3.empty:
        sns.violinplot(x='Epoch', y='Cell Count', data=df_ca2_ca3_ca3, ax=axs[1, 1], inner=None, color='lightcoral')
        sns.swarmplot(x='Epoch', y='Cell Count', data=df_ca2_ca3_ca3, ax=axs[1, 1], color='black', size=3)
        axs[1, 1].set_title('CA2-CA3 assemblies - CA3 cells count')

    plt.tight_layout()
    plt.show()


def get_above_threshold_neurons(patterns, pattern_types, ca2_indices, ca3_indices,ca1_indices, threshold):
    """
    Extract neurons above the threshold for each pattern type and epoch,
    and include pattern IDs for each assembly.
    """
    neurons_above_threshold = {}
    for epoch_name, pattern_set in patterns.items():
        neurons_above_threshold[epoch_name] = {'CA2 only': {}, 'CA3 only': {}, 'CA2-CA3': {}}
        
        all_indeces = ca2_indices + ca3_indices + ca1_indices

        a = pd.DataFrame(pattern_set.T, all_indeces)
        for pattern_type, assembly_indices in pattern_types[epoch_name].items():
            for j in assembly_indices:
                above_threshold_ser = a[j].where(a[j] > threshold)
                above_threshold = above_threshold_ser.dropna().index.tolist()
                if pattern_type == 'CA2 only' and any(neuron in ca2_indices for neuron in above_threshold):
                    neurons_above_threshold[epoch_name]['CA2 only'][j] = [neuron for neuron in above_threshold if neuron in ca2_indices]
                elif pattern_type == 'CA3 only' and any(neuron in ca3_indices for neuron in above_threshold):
                    neurons_above_threshold[epoch_name]['CA3 only'][j] = [neuron for neuron in above_threshold if neuron in ca3_indices]
                elif pattern_type == 'CA2-CA3':
                    neurons_above_threshold[epoch_name]['CA2-CA3'][j] = above_threshold
    return neurons_above_threshold