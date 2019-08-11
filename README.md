# DataFrameLabeler
A small ipywidget tool for labeling data frames inside jupyter.

# Installation

Currently, the only way to use the DataFrameLabeler is to clone this repositroy.

# Why?

This small tool was inspired by the fast.ai image cleaner widget https://docs.fast.ai/widgets.image_cleaner.html .
However, I needed a tool for tabular data.

# How to use?

```
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DataFrameLabeler import DataFrameLabeler

# If you have a pandas data frame where you want to assign each row eihter 'SUCCESS' or 'FAILURE'.
# like the following one.
length = 100
cols = ['A', 'B', 'C', 'D', 'E']
df = pd.DataFrame(np.random.rand(length, len(cols)), columns=cols)

# First you need a function responsible to print a single row.
def plotter(idx, row):
    fig = plt.figure()
    plt.plot([i for i in row[cols]])
    # plot should not be shown when called.
    plt.close(fig)
    return fig
    
# Afterwards, just construct a DataFrameLabeler object.
# If `target_col` exists in the data frame, its content will be used as preselection.
lbl = DataFrameLabeler(df,
                       labels=['FAILURE', 'SUCCESS'], # choices for the labels
                       plotter=plotter,               # function which plots each row
                       target_col='class_name',       # column name where the labels will be stored
                       width=3,                       # number of figures in each row
                       height=2                       # number of rows shown at once
                       )
```
![DataFrameLabeler](images/screenshot_lbl.png)
```
# To obtain the newly labeled data frame call lbl.get_labeled_data()
```
![Result](images/screenshot_res.png)

## TODO:
* rework how user defined plotter works, atm its horrifying, especially when
  using matplotlib
* proper styling of buttons
* allow groupby argument
* allow multi selection
* add automatic saving of intermediate result to csv or pickle file
* rethink interface
* add more unit tests
* Documentation
