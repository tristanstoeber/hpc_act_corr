# The Influence of Salient Experiences on Hippocampal Neural Assemblies

## Functions
1. **Process** Loading the dataset
2. **Assembly.py**   Lopes-dos-Santos et al (2013)
3. **Assembly_plot.py** For plotting assembly patterns in CA1,CA2,CA3 and Joint CA1-CA2 and CA2-CA3, assembly activity overtime, and plotting similarities between cell assemblies in different conditions

### Example
```
epoch_lengths = [
    len(A013_day7_bins_p['A013_day7_rest_hab_pre'].to_numpy()),
    len(A013_day7_bins_p['A013_day7_habituation_arena'].to_numpy()),
    len(A013_day7_bins_p['A013_day7_rest_hab_post'].to_numpy()),
    len(A013_day7_bins_p['A013_day7_habituation_cage'].to_numpy()),
    len(A013_day7_bins_p['A013_day7_rest_pre'].to_numpy()),
    len(A013_day7_bins_p['A013_day7_2novel_exposure'].to_numpy()),
    len(A013_day7_bins_p['A013_day7_exposure_reversed'].to_numpy()),
    len(A013_day7_bins_p['A013_day7_rest_post2'].to_numpy()),
    len(A013_day7_bins_p['A013_day7_1novel_exposure'].to_numpy()),
    len(A013_day7_bins_p['A013_day7_rest_post1'].to_numpy())
]

epoch_lengths

plot_assembly_activity_overtime(Social_joint_CA2_CA3_assembly_activities_over_time, epoch_lengths)

```

![activity](https://github.com/user-attachments/assets/0cee3168-0720-4884-9358-6b7a2bbe153b)


## Analysis

1. **Social_projectV7** _This version trace Social and Non-Scoial CA2-CA3 Joint cell Assemblies in A013_DAY7_

   Finding: a) **Social Joint Assembley have higher Activation rate in post sleep vs pre sleep and the difference decline over time**
            b) **Non Social Joint assemblies also have higher activation in post sleep**
            c) **Activation rates of both social and non-social joint assemblies during sleep following social stimulus were signficantly higher than following non-social experience**
3. **SocialProject_V9** _This version find joint assemblies in social experience epoch that have significant similarity with non_social epochs, and removes them, in order to isolate Social Joint cell assemblies_




