"""
"""
from collections.abc import Iterable
from typing import List, Callable, Union

from pandas import DataFrame
from numpy import nan
from ipywidgets import widgets, Layout
from IPython.display import clear_output, display

from Rowiter import Rowiter

class DataFrameLabeler():
    """Displays rows of Pandas data frame for labeling/relabeling.
    """
    def __init__(self, data: DataFrame, *,
                 label_col=None,
                 target_col=None,
                 labels: List=None,
                 plotter: Callable=None,
                 button_type: str='auto',
                 width=2,
                 height=2):
        """
        :param data: Pandas data frame where each row should be labeled.
        :param target_col: Selected labels will be written to this column.
               Will be created if not existent in `data`.
               Values in `target_col` will be used as preselection if parameter `label_col`
               is not set.
        :param labels: List of possible labels. If set additionally to `label_col` parameter,
               only labels in `labels` will be used.
        :param label_col: Column name in `data` where existing labels are stored which will
               be used as preselection when labeling the data. If `labels` is not set, all values
               in `label_col` will be used as labels.
        :param plotter: Callable which plots a row of the data frame.
               This function will be called with the index and the row of data which should be labeled.
        :param button_type: 'auto' - automatically select button type
                            'button' - use toggle buttons to select label
                            'list' - use selection list to select label
        :param width: Number of samples shown in one row.
        :param height: Number of rows shown in one batch.
        """
        self.data = data

        self.label_col = label_col
        if labels is not None:
            self.options = labels
        elif label_col is not None:
            if label_col not in self.data.columns:
                raise ValueError('`label_col` does not exists in `df`')
            self.options = self.data[label_col].unique().tolist()
        else:
            raise ValueError('either `labels` or `label_col` parameter must be set')

        if target_col is None:
            raise ValueError('`target_col` is necessary to save labels')
        else:
            self.target_col = target_col
            if not target_col in self.data.columns:
                self.data[target_col] = nan

        btypes = ['auto', 'button', 'list']
        if button_type not in btypes:
            raise ValueError('`button_type` must be one out of `' + '`, `'.join(btypes) + '`')

        if button_type == 'auto':
            if len(self.options) > 3:
                self.button_type = 'list'
            else:
                self.button_type = 'button'
        else:
            self.button_type = button_type

        # use simple row plotter if user does not provide plot function
        if plotter is not None:
            self.plotter = plotter
        else:
            def row_plotter(idx, row):
                print(idx)
                print(row)

            self.plotter = row_plotter


        self.rows = height
        self.columns = width
        self.batch_size = self.rows * self.columns
        self.outs = [widgets.Output() for i in range(self.batch_size)]

        # stores the active selection shown to the user
        self.active_selections = []

        self.rowiter = Rowiter(self.data)

        self.next_batch()
        self.render()


    def get_labeled_data(self) -> DataFrame:
        """ Return the labeled data frame."""
        return self.data

    def make_selection(self, rowiter) -> Union[widgets.ToggleButtons, widgets.Select]:
        """ Construct buttons/selection to choose a label"""
        row = rowiter[1]

        # set value of selector from label column if label column was set
        if self.label_col is not None and row[self.label_col] in self.options:
            value = row[self.label_col]
        else:
            # set value of selector from target column if the value
            # in the target column is one of the labels
            if row[self.target_col] in self.options:
                value = row[self.target_col]
            else:
                value = None

        if self.button_type == 'button':
            sel = widgets.ToggleButtons(
                    options=self.options,
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

    def make_next_button(self) -> widgets.Button:
        """Constructs the button to save the data and show the next batch."""
        if self.rowiter.distance_to_end() == 0:
            desc='Save'
            handler=self.handle_save
        else:
            desc='Save & Next'
            handler=self.next

        btn = widgets.Button(description=desc,
                             layout=Layout(width='70%', height='40px'))
        btn.on_click(handler)

        return btn

    def make_prev_button(self) -> widgets.Button:
        if self.rowiter.distance_to_begin() <= self.batch_size:
            desc='Save'
            handler=self.handle_save
        else:
            desc='Save & Back'
            handler=self.back

        btn = widgets.Button(description=desc,
                             layout=Layout(width='30%', height='40px'))
        btn.on_click(handler)

        return btn

    def make_nav_bar(self) -> widgets.HBox:
        widgetlist = []

        widgetlist.append(self.make_prev_button())
        widgetlist.append(self.make_next_button())

        return widgets.HBox(widgetlist)

    def save_selection(self):
        # go over current selection and save its state
        for rowiter, btn in self.active_selections:
            idx, _ = rowiter
            self.data.loc[idx, self.target_col] = btn.value if btn.value is not None else nan

    def handle_save(self, _):
        self.save_selection()

    def clear_selection(self):
        self.active_selections = []

    @classmethod
    def make_label_finished(cls) -> widgets.Label:
        """Constructs the label which shows that no more data has to be processed."""
        return widgets.Label("All data labeled")

    def next(self, _) -> None:
        """Prepares the next batch and render it"""

        self.save_selection()
        self.clear_selection()

        self.next_batch()
        self.render()

    def back(self, _) -> None:
        self.save_selection()
        self.clear_selection()

        self.prev_batch()
        self.render()

    def prev_batch(self):
        steps = self.batch_size + len(self.batch)
        self.rowiter.backward_until_first(steps=steps)

        self.next_batch()

    def next_batch(self):
        """Constructing the next batch by consuming self.rowiter"""
        self.batch = []
        try:
            # use all rows if we ignore target column
            for _ in range(self.batch_size):
                self.batch.append(next(self.rowiter))

        except StopIteration:
            pass

    def render(self) -> None:
        """Render output"""
        clear_output()

        if len(self.batch) != 0:
            widgetlist = [self.make_row(self.outs[i*self.columns:(i+1)*self.columns],
                                        self.batch[i*self.columns:(i+1)*self.columns])
                          for i in range(self.rows)]
            widgetlist.append(self.make_nav_bar())
            display(widgets.VBox(widgetlist))
        else:
            display(widgets.VBox([self.make_nav_bar()]))
