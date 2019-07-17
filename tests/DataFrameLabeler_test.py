import unittest
import sys

import pandas as pd
import numpy as np

from context import DataFrameLabeler as dfl


class DataFrameLabelerTestSuite(unittest.TestCase):
    def test_ctor(self):
        df = pd.DataFrame(np.arange(0, 10))

        # ctor needs target_col and either label_col or labels as argument
        self.assertRaises(ValueError, dfl.DataFrameLabeler, df)
        self.assertRaises(ValueError, dfl.DataFrameLabeler, df, target_col='target')
        self.assertRaises(ValueError, dfl.DataFrameLabeler, df, labels=['1', '2', '3'])
        # label column does not exists in df
        self.assertRaises(ValueError, dfl.DataFrameLabeler, df, label_col='label')


        labels=['1', '2', '3']
        lbl = dfl.DataFrameLabeler(df, target_col='target', labels=labels)

        # check that labels are set correctly
        self.assertEquals(lbl.options, labels)

        # check that target column is created
        self.assertIn('target', lbl.data.columns)

        # check that labels are extracted correctly from label_col
        df['label'] = np.random.choice(labels, df.shape[0])
        lbl = dfl.DataFrameLabeler(df, target_col='target', label_col='label')
        self.assertEquals(sorted(lbl.options), sorted(labels))



if __name__ == '__main__':
    unittest.main()
