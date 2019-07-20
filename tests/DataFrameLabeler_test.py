import unittest
import sys
from contextlib import contextmanager
from io import StringIO

import pandas as pd
import numpy as np

from context import DataFrameLabeler as dfl


@contextmanager
def capture_stdout():
    new_out = StringIO()
    old_out = sys.stdout
    try:
        sys.stdout = new_out
        yield sys.stdout
    finally:
        sys.stdout = old_out


class DataFrameLabelerTestSuite(unittest.TestCase):
    def test_ctor(self):
        df = pd.DataFrame(np.arange(0, 10))

        # capture output to not spoil test output
        with capture_stdout():
            # ctor needs target_col and either label_col or labels as argument
            self.assertRaises(ValueError, dfl.DataFrameLabeler, df)
            self.assertRaises(ValueError, dfl.DataFrameLabeler, df, target_col='target')
            self.assertRaises(ValueError, dfl.DataFrameLabeler, df, labels=['1', '2', '3'])
            # label column does not exists in df
            self.assertRaises(ValueError, dfl.DataFrameLabeler, df, label_col='label')


            labels=['1', '2', '3']
            lbl = dfl.DataFrameLabeler(df, target_col='target', labels=labels)


            # check that labels are set correctly
            self.assertEqual(lbl.options, labels)

            # check that target column is created
            self.assertIn('target', lbl.data.columns)

            # check that labels are extracted correctly from label_col
            df['label'] = np.random.choice(labels, df.shape[0])
            lbl = dfl.DataFrameLabeler(df, target_col='target', label_col='label')
            self.assertEqual(sorted(lbl.options), sorted(labels))


if __name__ == '__main__':
    unittest.main()
