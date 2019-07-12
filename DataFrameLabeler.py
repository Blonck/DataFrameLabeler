import pandas as pd
import numpy as np
from collections.abc import Iterable
from typing import List, Callable, Union

from ipywidgets import widgets, Layout
from IPython.display import clear_output, display


class DataFrameLabeler():
    """Displays rows of Pandas data frame for labeling/relabeling.

    TODO: 
    * allow multi selection
    * proper styling of buttons
    * allow groupby argument
    * add automatic saving of intermediate result to csv or pickle
    * currently the plotter function return value is put directly into an widgets.Output,
      it should also allow to return a widget
    * rethink interface
    """
    def __init__(self, data: pd.DataFrame, *,
                 label_col=None,
                 additional_labels: Iterable=None,
                 target_col=None,
                 overwrite_existing_targets=True,
                 labels: List=None,
                 plotter: Callable=None,
                 width=2,
                 height=2,
                 ):
        """
        :param: data Pandas data frame where each row should be labeled.
        :param: label_col Column name in `data` where existing labels are stored which will
                          be used as preselection when labeling the data.
                          Should not be set if `labels` parameter is set.
        :param: additional_labels Labels which should be used next to the ones in `label_col'
                                  Only valid if `label_col` is set.
        :param: overwrite_existing_targets If 'False' all rows which have a valid label in `target_col` will
                                           not be shown, and thus, not be relabeled.
        :param: labels List of possible labels.
                       Should not be set if `label_col` parameter is set.
        :param: plotter Callable which plots a row of the data frame.
                        This function will be called with the index and the row of data which should be labeled.
        :param: width Number of samples shown in one row.
        :param: height Number of rows shown in one batch.
        """
        self.data = data

        if plotter is None:
            raise ValueError('plotter argument must be set')

        # either use label_col or labels
        self.label_col = label_col
        if label_col is not None:
            self.options = self.data[label_col].unique().tolist()
        elif labels is not None:
            self.options = labels
        else:
            raise ValueError('Either label_col or labels must be set.')

        if target_col is None:
            raise ValueError('target_col is necessary to save labels')
        else:
            self.target_col = target_col
            if not target_col in self.data.columns:
                self.data[target_col] = np.nan

        self.overwrite = overwrite_existing_targets

        if additional_labels is not None:
            # throw out duplicated options
            self.options = list(set(self.options) + set(addiotional_labels))

        self.rows = height
        self.columns = width
        self.batch_size = self.rows * self.columns
        self.outs = [widgets.Output() for i in range(self.batch_size)]

        # stores the active selection shown to the user
        self.active_selections = []

        self.plotter = plotter
        self.it = self.data.iterrows()
        self.finished = False

        self.render()

    def get_labeled_data(self) -> pd.DataFrame:
        """ Return the labeled data frame."""
        return self.data

    def make_selection(self, rowiter) -> Union[widgets.ToggleButtons, widgets.Select]:
        """ Construct buttons/selection to choose a label"""
        row = rowiter[1]

        # set value of selector from label column if label column was set
        if self.label_col is not None:
            value = row[self.label_col]
        else:
            # set value of selector from target column if the value
            # in the target column is one of the labels
            if row[self.target_col] in self.options:
                value = row[self.target_col]
            else:
                value = None

        # use ToggleButton widget only the number of labels is small
        # TODO refactor: either make configurable or don't use magic number
        if len(self.options) <= 3:
            sel = widgets.ToggleButtons(
                    options=self.options,
                    orientation='horizontal',
                    value=value,
                    layout=Layout(width='100%'))
        else:
            sel =  widgets.Select(options=self.options,
                                  value=value)

        self.active_selections.append((rowiter, sel))
        return sel

    def make_box(self, out, rowiter) -> widgets.VBox:
        """Combines buttons to choose the label with the user defined plotter."""
        out.clear_output()
        with out:
            display(self.plotter(rowiter[0], rowiter[1]))
        return widgets.VBox([self.make_selection(rowiter), out])

    def make_row(self, outs, rowtuple) -> widgets.HBox:
        """Combines several boxes into a row."""
        widgetrow = [self.make_box(out, row) for out, row in zip(outs, rowtuple)]
        return widgets.HBox(widgetrow)

    def make_next_button(self, handler=None) -> widgets.Button:
        """Constructs the button to save the data and show the next batch."""
        btn = widgets.Button(description='Save & Next',
                             layout=Layout(width='100%', height='40px'))
        if handler is not None:
            btn.on_click(handler)
        return btn

    @classmethod
    def make_label_finished(cls) -> widgets.Label:
        """Constructs the label which shows that no more data has to be processed."""
        return widgets.Label("All data labeled")

    def next(self, _) -> None:
        """Prepares the next batch and render it"""

        # go over current selection and save its state
        for rowiter, btn in self.active_selections:
            idx, row = rowiter
            self.data.loc[idx, self.target_col] = btn.value if btn.value is not None else np.nan

        self.active_selections = []
        self.render()

    def next_batch(self):
        """Constructing the next batch by consuming self.it"""
        if self.finished:
            return

        self.batch = []
        try:
            # use all rows if we ignore target column
            if self.overwrite:
                for i in range(self.batch_size):
                    self.batch.append(next(self.it))
            # use the rows where the label in the target column is not one of the label
            else:
                count = 0
                while count < self.batch_size:
                    it = next(self.it)
                    idx, row = it
                    if(row[self.target_col] not in self.options):
                        self.batch.append(it)
                        count += 1
        except StopIteration:
            self.finished = True


    def render(self) -> None:
        """Render output"""
        clear_output()

        if self.finished:
            display(self.make_label_finished())
            return

        self.next_batch()

        if len(self.batch) != 0:
            widgetlist = [self.make_row(self.outs[i*self.columns:(i+1)*self.columns],
                                        self.batch[i*self.columns:(i+1)*self.columns])
                          for i in range(self.rows)]
            if self.finished:
                widgetlist.append(self.make_label_finished())
            else:
                widgetlist.append(self.make_next_button(handler=self.next))

            display(widgets.VBox(widgetlist))
        else:
            display(widgets.VBox([self.make_label_finished()]))
