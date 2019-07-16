import pandas as pd

class Rowiter():
    """Class allowing bidirectional iterating a pandas Frame.

    TODO: * input could also be a pd.Series

    Note: This concept makes not musch sense in the context of python iterators.
          Nevertheless, do it here until I have a better idea to abstract
          the forward and backward iteration away.
    """
    def __init__(self, df: pd.DataFrame):
        self.index = df.index
        self.df = df
        self.cur = 0
        self.length = len(self.index)

    def __iter__(self):
        return self

    def __next__(self):
        """Return the current element and iterate to the next row."""
        if self.cur >= self.length:
            raise StopIteration
        else:
            ret = self.index[self.cur]
            self.cur += 1
            return (ret, self.df.loc[ret])

    def forward(self, steps:int=1) -> None:
        """Iterate to the next row. Tak.

        :param steps: Iterate `steps` at once.
        """
        self.cur += steps
        self.cur = min(self.length, self.cur)

    def forward_until_last(self, steps:int=1) -> None:
        """Iterate to the next row, but stay at the last one if reached.

        :param steps: Iterate `steps` at once.
        """
        self.cur += steps
        self.cur = min(self.length-1, self.cur)

    def backward(self, steps:int=1) -> None:
        """Go back to the previous row.

        :param steps: Iterate `steps` at once.
        """
        self.cur -= steps
        self.cur = max(-1, self.cur)

    def backward_until_first(self, steps:int=1) -> None:
        """Go back to the previous row, but stay at the first one.

        :param steps: Iterate `steps` at once.
        """
        self.cur -= steps
        self.cur = max(0, self.cur)

    def distance_to_end(self) -> int:
        """Return the number of steps needed until the end of the data frame."""
        return self.length - self.cur

    def distance_to_begin(self) -> int:
        """Return the number of steps needed until the beginning of the data frame."""
        return self.cur

    def exceeded(self) -> bool:
        """Return 'True' if the iterator is over the bound in either of the two directions."""
        if self.cur < 0 or self.cur >= self.length:
            return True
        else:
            return False

    def get(self):
        """Return index and pd.Series of the current row.

        :raises: StopIteration if iterator is over the bound in either of the two directions.
        """
        if self.exceeded():
            raise StopIteration
        return (self.index[self.cur], self.df.loc[self.index[self.cur]])
