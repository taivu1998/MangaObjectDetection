import unittest

import pandas as pd

from manga_detection.data.splitting import split_dataframe


class SplittingTests(unittest.TestCase):
    def test_split_dataframe_lengths(self):
        df = pd.DataFrame({"value": range(10)})
        train_df, valid_df, test_df = split_dataframe(df, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, random_state=0)
        self.assertEqual(len(train_df), 8)
        self.assertEqual(len(valid_df), 1)
        self.assertEqual(len(test_df), 1)

    def test_split_dataframe_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            split_dataframe(pd.DataFrame(), random_state=0)


if __name__ == "__main__":
    unittest.main()
