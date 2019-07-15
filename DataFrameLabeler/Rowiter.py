import pandas as pd

class Rowiter():
    """Class allowing bidirectional iterating a pandas Frame.

    TODO: * input could also be a pd.Series

    Note: This concept makes no sense in the context of python iterators.
          Nevertheless, just do it here and see what happens.
    """
    def __init__(self, df: pd.DataFrame):
        self.index = df.index
        self.df = df
        self.cur = 0
        self.length = len(self.index)

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

    def distance_to_end(self) -> int:
        return self.length - self.cur

    def distance_to_begin(self) -> int:
        return self.cur

    def get(self):
        if self.end():
            raise StopIteration
        return (self.index[self.cur], self.df.loc[self.index[self.cur]])
