import unittest
import sys

import pandas as pd
import numpy as np

from context import Rowiter

class RowiterTestSuite(unittest.TestCase):
    def test_walkthrough(self):
        df = pd.DataFrame(np.arange(0, 10))
        self.assertEqual(df.shape, (10, 1))

        # check that normal walkthrough works
        rowiter = Rowiter.Rowiter(df)
        for idx, row in df.iterrows():
            tidx, trow = next(rowiter)
            self.assertEqual(idx, tidx)
            # checking series are the same
            self.assertTrue((row == trow).all())

        # next should lead to end of iteration
        self.assertRaises(StopIteration, next, rowiter)


    def test_forward(self):
        for steps in range(1, 5):
            with self.subTest(steps=steps):
                df = pd.DataFrame(np.arange(0, 10))
                self.assertEqual(df.shape, (10, 1))
                rowiter = Rowiter.Rowiter(df)

                for i in range(0, 10, steps):
                    tidx, trow = rowiter.get()
                    idx = df.index[i]
                    row = df.iloc[i, :]

                    self.assertEqual(idx, tidx)
                    self.assertTrue((row == trow).all())

                    rowiter.forward(steps)

                rowiter.forward(steps)
                self.assertRaises(StopIteration, rowiter.get)


    def test_forward_until_last(self):
        for steps in range(1, 5):
            with self.subTest(steps=steps):
                df = pd.DataFrame(np.arange(0, 10))
                self.assertEqual(df.shape, (10, 1))
                rowiter = Rowiter.Rowiter(df)

                for i in range(0, 10, steps):
                    tidx, trow = rowiter.get()
                    idx = df.index[i]
                    row = df.iloc[i, :]

                    self.assertEqual(idx, tidx)
                    self.assertTrue((row == trow).all())

                    rowiter.forward_until_last(steps)

                # stays at last element
                rowiter.forward_until_last(steps)
                tidx, trow = rowiter.get()
                idx = df.index[9]
                row = df.iloc[9, :]
                self.assertEqual(idx, tidx)
                self.assertTrue((row == trow).all())


    def test_backward(self):
        df = pd.DataFrame(np.arange(0, 10))
        self.assertEqual(df.shape, (10, 1))
        rowiter = Rowiter.Rowiter(df)

        # go to end of data frame
        for i in rowiter:
            pass

        # ensure that iterator is at end
        self.assertRaises(StopIteration, rowiter.get)

        # go backwards
        for i in range(9, -1, -1):
           rowiter.backward()

           tidx, trow = rowiter.get()
           idx = df.index[i]
           row = df.iloc[i, :]

           self.assertEqual(idx, tidx)
           self.assertTrue((row == trow).all())

        # should be no rows left
        rowiter.backward()
        self.assertRaises(StopIteration, rowiter.get)


    def test_backward_until_first(self):
        df = pd.DataFrame(np.arange(0, 10))
        self.assertEqual(df.shape, (10, 1))
        rowiter = Rowiter.Rowiter(df)

        # go to end of data frame
        for i in rowiter:
            pass

        # ensure that iterator is at end
        self.assertRaises(StopIteration, rowiter.get)

        # go backwards
        for i in range(9, -1, -1):
           rowiter.backward()

           tidx, trow = rowiter.get()
           idx = df.index[i]
           row = df.iloc[i, :]

           self.assertEqual(idx, tidx)
           self.assertTrue((row == trow).all())

        # stays at first element
        rowiter.backward_until_first()
        tidx, trow = rowiter.get()
        self.assertEqual(df.index[0], tidx)
        self.assertTrue((df.iloc[0, :] == trow).all())


    def test_distance(self):
        df = pd.DataFrame(np.arange(0, 10))
        self.assertEqual(df.shape, (10, 1))
        rowiter = Rowiter.Rowiter(df)

        for i in range(0, 10):
            self.assertAlmostEqual(10 - i, rowiter.distance_to_end())
            self.assertAlmostEqual(i, rowiter.distance_to_begin())
            rowiter.forward()


    def test_walkthrough_datetimeindex(self):
        length = 10
        index = pd.date_range(start='1/1/2019', periods=length)
        df = pd.DataFrame(np.arange(0, length), index=index)
        self.assertEqual(df.shape, (length, 1))

        # check that normal walkthrough works
        rowiter = Rowiter.Rowiter(df)
        for idx, row in df.iterrows():
            tidx, trow = next(rowiter)
            self.assertEqual(idx, tidx)
            # checking series are the same
            self.assertTrue((row == trow).all())

        # next should lead to end of iteration
        self.assertRaises(StopIteration, next, rowiter)


    def test_walkthrough_multiindex(self):
        iterables = [['bar', 'baz', 'foo', 'qux'], ['one', 'two']]
        index = pd.MultiIndex.from_product(iterables, names=['first', 'second'])

        df = pd.DataFrame()
        np.random.seed(42)
        df['1'] = pd.Series(np.random.randn(8), index=index)
        df['2'] = pd.Series(np.random.randn(8), index=index)

        # check that normal walkthrough works
        rowiter = Rowiter.Rowiter(df)
        for idx, row in df.iterrows():
            tidx, trow = next(rowiter)
            self.assertEqual(idx, tidx)
            # checking series are the same
            self.assertTrue((row == trow).all())

        # next should lead to end of iteration
        self.assertRaises(StopIteration, next, rowiter)


if __name__ == '__main__':
    unittest.main()
