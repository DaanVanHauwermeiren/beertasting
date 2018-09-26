from __future__ import print_function, division
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from time import strftime, localtime
import json
import os
# Make it work for Python 2+3 and with Unicode
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

sns.set_style('darkgrid')
sns.set_context(context='notebook', font_scale=1.3)
fivethirtyeight = [
    "#30a2da",
    "#fc4f30",
    "#e5ae38",
    "#6d904f",
    "#8b8b8b"
  ]
sns.set_palette(fivethirtyeight)
# Set the default color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', fivethirtyeight)
plt.rcParams['figure.figsize'] = [9,6]  #3/2 


class Beertasting():
    """beertasting class
    """

    def __init__(self, *args, **kwargs):
        # args -- tuple of anonymous arguments
        # kwargs -- dictionary of named arguments
        if kwargs.get('dummyinit', False):
            print("initialising the tasting magic with dummy values for"
                  "testing purposes!")
            self._alltastings = [
                {"taster": "Person_1", "beer": "Beer_A", "score": 3.0, "hoppig": 1.5, "kleur": "blond"},
                {"taster": "Person_1", "beer": "Beer_A", "score": 4.0, "hoppig": 3.5, "kleur": "blond"},
                {"taster": "Person_1", "beer": "Beer_B", "score": 5.0, "hoppig": 2.0, "kleur": "hoogblond"},
                {"taster": "Person_1", "beer": "Beer_B", "score": 3.0, "hoppig": 4.5, "kleur": "blond"},
                {"taster": "Person_1", "beer": "Beer_C", "score": 5.0, "hoppig": 1.0, "kleur": "blond"},
                {"taster": "Person_2", "beer": "Beer_A", "score": 3.0, "hoppig": 4.0, "kleur": "bruin"},
                {"taster": "Person_2", "beer": "Beer_A", "score": 2.0, "hoppig": 3.0, "kleur": "blond"},
                {"taster": "Person_2", "beer": "Beer_B", "score": 4.0, "hoppig": 2.5, "kleur": "blond"},
                {"taster": "Person_2", "beer": "Beer_C", "score": 3.5, "hoppig": 5.0, "kleur": "blond"},
                {"taster": "Person_2", "beer": "Beer_D", "score": 5.0, "hoppig": 2.0, "kleur": "rood"},
                {"taster": "Person_3", "beer": "Beer_A", "score": 2.5, "hoppig": 1.5, "kleur": "bruin"},
                {"taster": "Person_3", "beer": "Beer_A", "score": 1.0, "hoppig": 3.5, "kleur": "bruin"},
                {"taster": "Person_3", "beer": "Beer_A", "score": 2.0, "hoppig": 4.0, "kleur": "hoogblond"},
                {"taster": "Person_3", "beer": "Beer_A", "score": 3.5, "hoppig": 2.0, "kleur": "blond"},
                {"taster": "Person_3", "beer": "Beer_B", "score": 4.0, "hoppig": 4.5, "kleur": "blond"},
                {"taster": "Person_3", "beer": "Beer_B", "score": 5.0, "hoppig": 3.0, "kleur": "rood"},
                {"taster": "Person_3", "beer": "Beer_C", "score": 1.0, "hoppig": 1.0, "kleur": "bruin"},
                {"taster": "Person_3", "beer": "Beer_D", "score": 4.0, "hoppig": 4.0, "kleur": "blond"},
            ]
            self._generate_datatable()
        else:
            print("initialising an empty tasting object!")
            self._alltastings = []

    def _generate_datatable(self):
        self._datatable = pd.DataFrame(data=self._alltastings)
        self._datatable.set_index(["beer", "taster"], inplace=True)

    @property
    def alltastings(self):
        return getattr(self, '_alltastings', [])

    @property
    def datatable(self):
        return getattr(self, '_datatable', None)

    def get_tested_beers(self):
        """return a list of all the tested beers
        """
        return self._datatable.index.get_level_values(0).unique().tolist()
    
    def get_tasters(self):
        """return a list of all the tasters
        """
        return self._datatable.index.get_level_values(1).unique().tolist()

    def get_tested_attributes(self, includetype=False):
        """return a list of all the tested attributes
        """
        if includetype:
            return self.datatable.dtypes.replace(to_replace={
                float: "numerical",
                int: "numerical",
                object: "categorical"
            })
        else:
            return self._datatable.columns.tolist()

    def save_tasting(self):
        """save the current tastings in
            ./stored_tastings/TIMESTAMP_entries.json
            ./stored_tastings/TIMESTAMP_dataframe.csv
        """
        timestamp = strftime("%Y_%m_%d__%H:%M:%S", localtime())
        basefolder = "./stored_tastings/"
        # Write JSON file
        with io.open(basefolder + timestamp + "_entries.json", 'w', encoding='utf8') as outfile:
            str_ = json.dumps(self.alltastings,
                              indent=4, sort_keys=True,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))
        # write csv file
        self.datatable.to_csv(basefolder + timestamp + "_dataframe.csv")
        print("saved the tastings!")

    def load_tasting(self, basefilepath, overwrite=False):
        """load previously stored tasting data
        
        Arguments:
            basefilepath {str} -- str of a file path, e.g.
            "./stored_tastings/2018_07_01__10:01:33"
        """
        print("loading previous tasting!")
        assert os.path.isfile(basefilepath + "_entries.json"), "json file not found"
        assert os.path.isfile(basefilepath + "_dataframe.csv"), "dataframe not found"
        if (len(self.alltastings) != 0) & overwrite:
            print("warning: data is present, overwriting data")
        elif (len(self.alltastings) != 0) & (not overwrite):
            raise Exception("Data is present, use option overwrite=True to force loading")
        # Read JSON file
        with open(basefilepath + "_entries.json") as data_file:
            self._alltastings = json.load(data_file)
        # read dataframe
        self._datatable = pd.read_csv(basefilepath + "_dataframe.csv", index_col=[0,1])

    def taste_beer(self, tastingdict):
        """add a single tasting review from one person

        Arguments:
            tastingdict {dict} -- the dict with information on the tasting,
            contains at least: taster, beer, and score as keys
        """
        if "taster" not in tastingdict.keys():
            raise Exception("please include taster in the tastingdict")
        if "beer" not in tastingdict.keys():
            raise Exception("please include beer in the tastingdict")
        if "score" not in tastingdict.keys():
            raise Exception("please include score in the tastingdict")

        self._alltastings.append(tastingdict)
        self._generate_datatable()

    def plot_beer_summary(self, beer):
        """plot a summary of the beer scores, and numerical attributes

        Arguments:
            beers {str} -- beer name to indentify what to plot
        """
        fig, ax = plt.subplots()
        data = self.datatable.loc[beer, :]
        sns.violinplot(data=data, ax=ax)
        ax.set_title(beer)
        # auto rotate axis labels
        fig.autofmt_xdate()
        ax.set_ylim(0,5)
        return fig, ax

    def plot_beer_summary_all(self):
        """call the function plot_beer_summary for all beers
        """
        allbeers = self.datatable.index.get_level_values(0).unique()
        return [self.plot_beer_summary(beer) for beer in allbeers]

    def plot_beer_attribute(self, attribute, splitbytaster=False):
        """plot the values of the attribute for all the beers
        
        Arguments:
            attribute {str} -- the type of attribute to plot, e.g. score,
            hoppig ...
        """
        assert attribute in self.datatable.columns, "attribute " + attribute + ""\
        " has not been tested"
        assert pd.api.types.is_numeric_dtype(self.datatable[attribute]), "select"\
        "an attribute which is numerical"
        if splitbytaster:
            hue = "taster"
        else:
            hue = None
        fig, ax = plt.subplots()
        sns.violinplot(x="beer", y=attribute,
                       data=self.datatable.reset_index(), hue=hue, ax=ax)
        # auto rotate axis labels
        fig.autofmt_xdate()
        # manually set rotation:
        """for tick in ax.get_xticklabels():
            tick.set_rotation(60)"""
        return fig, ax

    def plot_beer_categorical_attribute(self, attribute, splitbytaster=False):
        """plot the values of the categorical attribute for all the beers
        
        Arguments:
            attribute {str} -- the type of categorical attribute to plot, e.g. kleur,
            hoppig ...
        """
        assert attribute in self.datatable.columns, "attribute " + attribute + ""\
        " has not been tested"
        assert (
            (not pd.api.types.is_numeric_dtype(self.datatable[attribute]))
            & (pd.api.types.is_string_dtype(self.datatable[attribute]))
            ), "select an attribute that is categorical, not numerical"
        fig, ax = plt.subplots()
        data = pd.get_dummies(self.datatable[attribute]).groupby(["beer"]).sum()
        data.plot(marker='o', ax=ax)
        ax.set_title(attribute)
        return fig, ax

    def _calcfigwidth(self, height, fcolorbar=0.05):
        ''' calculates the width of the figure, based on the height, and the size fraction
        of the subfigure (colorbar) compared to the contourplot.
        The width is calculated as such that the resulting plot is square (i.e. the x and y
        axis are equal in length)
        fcolorbar is the size fraction of the colorbar compared to the figure
        '''
        wspace = plt.rcParams['figure.subplot.wspace']
        left = plt.rcParams['figure.subplot.left']
        right = plt.rcParams['figure.subplot.right']
        top = plt.rcParams['figure.subplot.top']
        bottom = plt.rcParams['figure.subplot.bottom']
        figwidth = (top - bottom)*height/(((1-left-(1-right))/(1+fcolorbar+wspace*(1-fcolorbar)/2)))
        return figwidth

    def plot_heatmap(self, taster=None, figheight=None, fcolorbar=.05):
        """[summary]
        
        Arguments:
            taster {[type]} -- [description]
        
        Keyword Arguments:
            figheight {[type]} -- [description] (default: {None})
            fcolorbar {float} -- [description] (default: {.05})
        """

        if figheight is None:
            figheight = plt.rcParams['figure.figsize'][1]
        figwidth = self._calcfigwidth(figheight, fcolorbar)
        fig, (ax, cax) = plt.subplots(ncols=2, figsize=(figwidth, figheight),
                                    gridspec_kw={"width_ratios":[1, fcolorbar]})
        
        if taster:
            data_corr = self.datatable.reset_index(level=0).loc[taster].corr()
            title = "taste profile of " + taster
        else:
            data_corr = self.datatable.corr()
            title = "general taste profile"
        # Generate a mask for the upper triangle
        mask = np.zeros_like(data_corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(data=data_corr, mask=mask, center=0, vmax=1, cmap="BrBG",
                    annot=True, ax=ax, cbar_ax=cax)
        ax.set_title(title)
        return fig, ax, cax

    def plot_regression(self, x_attribute, y_attribute, splitbytaster=False,
                        order=1):
        data = self.datatable.reset_index()
        assert x_attribute in data.columns, "attribute " + x_attribute + ""\
        " has not been tested"
        assert y_attribute in data.columns, "attribute " + y_attribute + ""\
        " has not been tested"
        if splitbytaster:
            hue = "taster"
        else:
            hue = None
        g = sns.lmplot(x_attribute, y_attribute, data=data, hue=hue,
                       order=order,
                       x_jitter=.15, y_jitter=.15) 
        g.set(xlim=(0,5), ylim=(0,5))
        return g

