import pandas as pd
import numpy as np
from collections.abc import Iterable
from typing import List, Callable, Union

from ipywidgets import widgets, Layout
from IPython.display import clear_output, display


class rowiter():
    """Class allowing bidirectional iterating a pandas Frame.

    When the calling next/previous on end, this iterator raise StopIteration,
    but stays at its current position.

    Note: This concept makes no sense in the context of python iterators.
          Nevertheless, just do it here and see what happens.
    """
    def __init__(self, df: pd.DataFrame):
        self.index = df.index
        self.df = df
        self.cur = 0
        self.length = len(self.index)
        if self.length == 0:
            raise StopIteration

    def __next__(self):
        if self.cur >= self.length:
            raise StopIteration
        else:
            ret = self.index[self.cur]
            self.cur += 1
            return (ret, self.df.loc[ret])

    def forward(self, steps:int=1) -> None:
        self.cur += steps
        self.cur = min(self.length, self.cur)

    def forward_until_last(self, steps:int=1) -> None:
        self.cur += steps
        self.cur = min(self.length-1, self.cur)

    def backward(self, steps:int=1) -> None:
        self.cur -= steps
        self.cur = max(-1, self.cur)

    def backward_until_first(self, steps:int=1) -> None:
        self.cur -= steps
        self.cur = max(0, self.cur)

    def end(self) -> bool:
        if self.cur < 0 or self.cur >= self.length:
            return True
        else:
            return False

    def is_first(self) -> bool:
        return self.cur == 0

    def is_last(self) -> bool:
        return self.cur == (self.length - 1)

    def get(self):
        if self.end:
            raise StopIteration
        return (self.index[self.cur], self.df.loc[self.index[self.cur]])

    def get_state(self) -> int:
        return self.cur

    def set_state(self, state: int) -> None:
        self.cur = state


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
    * allow going back
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
            raise ValueError('`plotter` argument must be set')

        # either use label_col or labels
        self.label_col = label_col
        if label_col is not None:
            self.options = self.data[label_col].unique().tolist()
        elif labels is not None:
            self.options = labels
        else:
            raise ValueError('Either `label_col` or `labels` must be set.')

        if target_col is None:
            raise ValueError('`target_col` is necessary to save labels')
        else:
            self.target_col = target_col
            if not target_col in self.data.columns:
                self.data[target_col] = np.nan

        if additional_labels is not None:
            # throw out duplicated options
            self.options = list(set(self.options) + set(addiotional_labels))

        self.overwrite = overwrite_existing_targets

        # the index which rows should not be touched
        if not self.overwrite:
            self.ignore_row = self.data[target_col].isin(self.options)


        self.rows = height
        self.columns = width
        self.batch_size = self.rows * self.columns
        self.outs = [widgets.Output() for i in range(self.batch_size)]

        # stores the active selection shown to the user
        self.active_selections = []

        self.plotter = plotter
        self.it = self.data.iterrows()
        self.last_frame = False
        self.first_frame = True

        self.rowiter = rowiter(self.data)

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
                             layout=Layout(width='70%', height='40px'))
        if handler is not None:
            btn.on_click(handler)
        return btn

    def make_prev_button(self, handler=None) -> widgets.Button:
        btn = widgets.Button(description='Save & Back',
                             layout=Layout(width='30%', height='40px'))
        if handler is not None:
            btn.on_click(handler)
        return btn

    def make_nav_bar(self) -> widgets.HBox:
        widgetlist = []

        widgetlist.append(self.make_prev_button())
        widgetlist.append(self.make_next_button(handler=self.next))

        return widgets.HBox(widgetlist)


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


    def back(self, _) -> None:
        pass

    def next_batch(self):
        """Constructing the next batch by consuming self.rowiter"""
        if self.last_frame:
            return

        self.batch = []
        try:
            # use all rows if we ignore target column
            if self.overwrite:
                for i in range(self.batch_size):
                    self.batch.append(next(self.rowiter))
            # use the rows where the label in the target column is not one of the label
            else:
                count = 0
                while count < self.batch_size:
                    it = next(self.rowiter)
                    idx, row = it
                    if(not self.ignore_row.loc[idx]):
                        self.batch.append(it)
                        count += 1
        except StopIteration:
            self.last_frame = True

    def render(self) -> None:
        """Render output"""
        clear_output()

        if self.last_frame:
            display(widgets.VBox([self.make_label_finished(),
                                  self.make_nav_bar()]))
            return

        self.next_batch()

        if len(self.batch) != 0:
            widgetlist = [self.make_row(self.outs[i*self.columns:(i+1)*self.columns],
                                        self.batch[i*self.columns:(i+1)*self.columns])
                          for i in range(self.rows)]
            widgetlist.append(self.make_nav_bar())
            display(widgets.VBox(widgetlist))
        else:
            display(widgets.VBox([self.make_nav_bar()]))
