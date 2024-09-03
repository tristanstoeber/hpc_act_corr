import numpy as np
import nelpy as nel
from scipy import stats
import os
import scipy.io

import logging
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
import assembly
from typing import Tuple, Union
from import_data import (
     load_cell_metrics,
     load_epochs, 
     load_SleepState_states,
     load_spikes,
     load_ripples_events,
     event_triggered_average_fast
     )

class AssemblyReact(object):
    """
    Class for running assembly reactivation analysis

    Core assembly methods come from assembly.py by Vítor Lopes dos Santos
        https://doi.org/10.1016/j.jneumeth.2013.04.010

    Parameters:
    -----------
    basepath: str
        Path to the session folder
    brainRegion: str
        Brain region to restrict to. Can be multi ex. "CA1|CA2"
    putativeCellType: str
        Cell type to restrict to
    weight_dt: float
        Time resolution of the weight matrix
    z_mat_dt: float
        Time resolution of the z matrix
    method: str
        Defines how to extract assembly patterns (ica,pca).
    nullhyp: str
        Defines how to generate statistical threshold for assembly detection (bin,circ,mp).
    nshu: int
        Number of shuffles for bin and circ null hypothesis.
    percentile: int
        Percentile for mp null hypothesis.
    tracywidom: bool
        If true, uses Tracy-Widom distribution for mp null hypothesis.

    attributes:
    -----------
    st: spike train (nelpy:SpikeTrainArray)
    cell_metrics: cell metrics (pandas:DataFrame)
    ripples: ripples (nelpy:EpochArray)
    patterns: assembly patterns (numpy:array)
    assembly_act: assembly activity (nelpy:AnalogSignalArray)

    methods:
    --------
    load_data: load data (st, ripples, epochs)
    restrict_to_epoch: restrict to a epoch
    get_z_mat: get z matrix
    get_weights: get assembly weights
    get_assembly_act: get assembly activity
    n_assemblies: number of detected assemblies
    isempty: isempty (bool)
    copy: returns copy of class
    plot: stem plot of assembly weights
    find_members: find members of an assembly

    *Usage*::

        >>> # create the object assembly_react
        >>> assembly_react = assembly_reactivation.AssemblyReact(
        ...    basepath=basepath,
        ...    )

        >>> # load need data (spikes, ripples, epochs)
        >>> assembly_react.load_data()

        >>> # detect assemblies
        >>> assembly_react.get_weights()

        >>> # visually inspect weights for each assembly
        >>> assembly_react.plot()

        >>> # compute time resolved signal for each assembly
        >>> assembly_act = assembly_react.get_assembly_act()

        >>> # locate members of assemblies
        >>> assembly_members = assembly_react.find_members()

    """

    def __init__(
        self,
        project_path: Union[str, None] = None,
        mice_name: Union[str, None] = None,
        brainRegion: str = "CA1",
        putativeCellType: str = "Pyramidal Cell",
        weight_dt: float = 0.025,
        z_mat_dt: float = 0.002,
        method: str = "ica",
        nullhyp: str = "mp",
        nshu: int = 1000,
        percentile: int = 99,
        tracywidom: bool = False,
        whiten: str = "unit-variance",
    ):
        self.project_path = project_path
        self.mice_name = mice_name
        self.brainRegion = brainRegion
        self.putativeCellType = putativeCellType
        self.weight_dt = weight_dt
        self.z_mat_dt = z_mat_dt
        self.method = method
        self.nullhyp = nullhyp
        self.nshu = nshu
        self.percentile = percentile
        self.tracywidom = tracywidom
        self.whiten = whiten
        self.type_name = self.__class__.__name__


    def add_st(self, st):
        self.st = st


    def add_epoch_df(self, epoch_df):
        self.epoch_df = epoch_df

    def load_spikes(self):
        """
        loads spikes from the session folder
        """
        self.st , self.cell_metrics = load_spikes(
            self.project_path,
            self.mice_name,
            brainRegion=self.brainRegion,
            putativeCellType=self.putativeCellType,
            support=self.epochs,
        )
        
        
           
    def load_ripples(self):
        """
        loads ripples from the session folder
        """
        ripples = load_ripples_events(self.project_path,self.mice_name)
        self.ripples = nel.EpochArray(
            [np.array([ripples.start, ripples.stop]).T], domain=self.time_support
        )

    def load_epoch(self):
        """
        loads epochs from the session folder
        """
        epoch_df = load_epochs(self.project_path,self.mice_name)
        self.time_support = nel.EpochArray(
            [epoch_df['Start_Time'].iloc[0],epoch_df['End_Time'].iloc[-1]]
        )
        self.epochs = nel.EpochArray(
            epoch_df[['Start_Time', 'End_Time']].values,
            domain=self.time_support,
        )
        self.epoch_df = epoch_df

    def load_data(self):
        """
        loads data (spikes,ripples,epochs) from the session folder
        """
        self.load_epoch()
        self.load_spikes()
        self.load_ripples()

    def restrict_epochs_to_pre_task_post(self) -> None:
        """
        Restricts the epochs to the specified epochs
        """
        # fetch data
        epoch_df = load_epochs(self.project_path,self.mice_name)
        # compress back to back sleep epochs (an issue further up the pipeline)
        # restrict to pre task post epochs

        self.epoch_df = epoch_df.iloc[[4,5,7]]
        # convert to epoch array and add to object
        self.epochs = nel.EpochArray(
            self.epoch_df[['Start_Time', 'End_Time']].values,
            label="session_epochs",
            domain=self.time_support,
        )

    def restrict_to_epoch(self, epoch):
        """
        Restricts the spike data to a specific epoch
        """
        self.st_resticted = self.st[epoch]



    def get_z_mat(self, st):
        """
        To increase the temporal resolution beyond the
        bin-size used to identify the assembly patterns,
        z(t) was obtained by convolving the spike-train
        of each neuron with a kernel-function
        """
        # binning the spike train
        z_t = st.bin(ds=self.z_mat_dt)
        # gaussian kernel to match the bin-size used to identify the assembly patterns
        sigma = self.weight_dt / np.sqrt(int(1000 * self.weight_dt / 2))
        z_t.smooth(sigma=sigma, inplace=True)
        # zscore the z matrix
        z_scored_bst = stats.zscore(z_t.data, axis=1)
        # make sure there are no nans, important as strengths will all be nan otherwise
        z_scored_bst[np.isnan(z_scored_bst).any(axis=1)] = 0

        return z_scored_bst, z_t.bin_centers

    def get_weights(self, epoch=None):
        """
        Gets the assembly weights
        """

        # check if st has any neurons
        if self.st.isempty:
            self.patterns = []
            return

        if epoch is not None:
            bst = self.st[epoch].bin(ds=self.weight_dt).data
        else:
            bst = self.st.bin(ds=self.weight_dt).data

        if (bst == 0).all():
            self.patterns = []
        else:
            self.patterns, _, _ = assembly.runPatterns(
                bst,
                method=self.method,
                nullhyp=self.nullhyp,
                nshu=self.nshu,
                percentile=self.percentile,
                tracywidom=self.tracywidom,
                whiten=self.whiten,
            )

    def get_assembly_act(self, epoch=None):
        # check for num of assemblies first
        if self.n_assemblies() == 0:
            return nel.AnalogSignalArray(empty=True)

        if epoch is not None:
            zactmat, ts = self.get_z_mat(self.st[epoch])
        else:
            zactmat, ts = self.get_z_mat(self.st)

        assembly_act = nel.AnalogSignalArray(
            data=list(assembly.computeAssemblyActivity(self.patterns, zactmat)),
            timestamps=ts,
            fs=1 / self.z_mat_dt,
        )
        return assembly_act

    def plot(
        self,
        plot_members: bool = True,
        central_line_color: str = "grey",
        marker_color: str = "k",
        member_color: Union[str, list] = "#6768ab",
        line_width: float = 1.25,
        markersize: float = 4,
        x_padding: float = 0.2,
        figsize: Union[tuple, None] = None,
    ):
        """
        plots basic stem plot to display assembly weights
        """

        if not hasattr(self, "patterns"):
            return f"run get_weights first"
        else:
            # if self.patterns == []:
            #     return None, None
            if plot_members:
                self.find_members()
            if figsize is None:
                if self.n_assemblies() == 1:
                    figsize = (self.n_assemblies() + 1, np.round(self.n_assemblies() / 1))
                else:
                    
                    figsize = (self.n_assemblies() + 1, np.round(self.n_assemblies() / 2))
            # set up figure with size relative to assembly matrix
            fig, axes = plt.subplots(
                1,
                self.n_assemblies(),
                figsize=figsize,
                sharey=True,
                sharex=True,
            )
            # iter over each assembly and plot the weight per cell
            for i in range(self.n_assemblies()):
                markerline, stemlines, baseline = axes[i].stem(
                    self.patterns[i, :], orientation="horizontal"
                )
                markerline._color = marker_color
                baseline._color = central_line_color
                baseline.zorder = -1000
                plt.setp(stemlines, "color", plt.getp(markerline, "color"))
                plt.setp(stemlines, linewidth=line_width)
                plt.setp(markerline, markersize=markersize)

                if plot_members:
                    current_pattern = self.patterns[i, :].copy()
                    current_pattern[~self.assembly_members[i, :]] = np.nan
                    markerline, stemlines, baseline = axes[i].stem(
                        current_pattern, orientation="horizontal"
                    )
                    if isinstance(
                        member_color, sns.palettes._ColorPalette
                    ) or isinstance(member_color, list):
                        markerline._color = member_color[i]
                    else:
                        markerline._color = member_color
                    baseline._color = "#00000000"
                    baseline.zorder = -1000
                    plt.setp(stemlines, "color", plt.getp(markerline, "color"))
                    plt.setp(stemlines, linewidth=line_width)
                    plt.setp(markerline, markersize=markersize)

                axes[i].spines["top"].set_visible(False)
                axes[i].spines["right"].set_visible(False)

            # give room for marker
            axes[0].set_xlim(
                -self.patterns.max() - x_padding, self.patterns.max() + x_padding
            )

            axes[0].set_ylabel("Neurons #")
            axes[0].set_xlabel("Weights (a.u.)")

            return fig, axes

    def n_assemblies(self):
        if hasattr(self, "patterns"):
            # if self.patterns == []:
            #     return 0
            # elif self.patterns is None:
            #     return 0
            return self.patterns.shape[0]

    @property
    def isempty(self):
        if hasattr(self, "st"):
            return False
        elif not hasattr(self, "st"):
            return True

    def copy(self):
        """Returns a copy of the current class."""
        newcopy = copy.deepcopy(self)
        return newcopy

    def __repr__(self) -> str:
        if self.isempty:
            return f"<{self.type_name}: empty>"

        # if st data as been loaded and patterns have been computed
        if hasattr(self, "patterns"):
            n_units = f"{self.st.n_active} units"
            n_patterns = f"{self.n_assemblies()} assemblies"
            dstr = f"of length {self.st.support.length}"
            return "<%s: %s, %s> %s" % (self.type_name, n_units, n_patterns, dstr)

        # if st data as been loaded
        if hasattr(self, "st"):
            n_units = f"{self.st.n_active} units"
            dstr = f"of length {self.st.support.length}"
            return "<%s: %s> %s" % (self.type_name, n_units, dstr)

    def find_members(self) -> np.ndarray:
        """
        Finds significant assembly patterns and signficant assembly members

        Output:
            assembly_members: a ndarray of booleans indicating whether each unit is a significant member of an assembly

        also, sets self.assembly_members and self.valid_assembly

        self.valid_assembly: a ndarray of booleans indicating an assembly has members with the same sign (Boucly et al. 2022)

        """

        def Otsu(vector: np.ndarray) -> Tuple[np.ndarray, float, float]:
            """
            The Otsu method for splitting data into two groups.
            This is somewhat equivalent to kmeans(vector,2), but while the kmeans implementation
            finds a local minimum and may therefore produce different results each time,
            the Otsu implementation is guaranteed to find the best division every time.

            input:
                vector: arbitrary vector
            output:
                group: binary class
                threshold: threshold used for classification
                em: effectiveness metric

            From Raly
            """
            sorted = np.sort(vector)
            n = len(vector)
            intraClassVariance = [np.nan] * n
            for i in np.arange(n):
                p = (i + 1) / n
                p0 = 1 - p
                if i + 1 == n:
                    intraClassVariance[i] = np.nan
                else:
                    intraClassVariance[i] = p * np.var(sorted[0 : i + 1]) + p0 * np.var(
                        sorted[i + 1 :]
                    )

            minIntraVariance = np.nanmin(intraClassVariance)
            idx = np.nanargmin(intraClassVariance)
            threshold = sorted[idx]
            group = vector > threshold

            em = 1 - (minIntraVariance / np.var(vector))

            return group, threshold, em

        is_member = []
        keep_assembly = []
        for pat in self.patterns:
            isMember, _, _ = Otsu(np.abs(pat))
            is_member.append(isMember)

            if np.any(pat[isMember] < 0) & np.any(pat[isMember] > 0):
                keep_assembly.append(False)
            elif sum(isMember) == 0:
                keep_assembly.append(False)
            else:
                keep_assembly.append(True)

        self.assembly_members = np.array(is_member)
        self.valid_assembly = np.array(keep_assembly)

        return self.assembly_members
