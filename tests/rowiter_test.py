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





if __name__ == '__main__':
    unittest.main()
