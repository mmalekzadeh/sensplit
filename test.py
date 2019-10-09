import unittest
import pandas as pd
from sensplit import dataframe_splitter
from pandas.util.testing import assert_frame_equal

class DataFrameSplitterTest(unittest.TestCase):
    def __init__(self,  *args, **kwargs):
        super(DataFrameSplitterTest, self).__init__(*args, **kwargs)        
        # self.dataset = np.repeat(np.arange(21).reshape(3, 7),4, axis=0)
        # np.random.RandomState(0).shuffle(self.dataset)
        self.dataset = pd.DataFrame([
                                    [ 7,  8,  9, 10, 11, 12, 13],
                                    [14, 15, 16, 17, 18, 19, 20],
                                    [ 7,  8,  9, 10, 11, 12, 13],
                                    [14, 15, 16, 17, 18, 19, 20],
                                    [ 0,  1,  2,  3,  4,  5,  6],
                                    [14, 15, 16, 17, 18, 19, 20],
                                    [ 0,  1,  2,  3,  4,  5,  6],
                                    [ 7,  8,  9, 10, 11, 12, 13],
                                    [14, 15, 16, 17, 18, 19, 20],
                                    [ 0,  1,  2,  3,  4,  5,  6],
                                    [ 0,  1,  2,  3,  4,  5,  6],
                                    [ 7,  8,  9, 10, 11, 12, 13]
                                    ], columns=list('ABCDEFG'))

        self.trial_train_data = pd.DataFrame([
                                    [7, 8, 9, 10, 11, 12, 13],
                                    [7, 8, 9, 10, 11, 12, 13],
                                    [7, 8, 9, 10, 11, 12, 13],
                                    [7, 8, 9, 10, 11, 12, 13],
                                    [14, 15, 16, 17, 18, 19, 20],
                                    [14, 15, 16, 17, 18, 19, 20],
                                    [14, 15, 16, 17, 18, 19, 20],
                                    [14, 15, 16, 17, 18, 19, 20]
                                    ],
                                    columns=list('ABCDEFG'))

        self.trial_test_data = pd.DataFrame([
                                    [0, 1, 2, 3, 4, 5, 6],
                                    [0, 1, 2, 3, 4, 5, 6],
                                    [0, 1, 2, 3, 4, 5, 6],
                                    [0, 1, 2, 3, 4, 5, 6]],
                                    columns=list('ABCDEFG'))
        
        self.ratio_train_data = pd.DataFrame([
                                    [0, 1, 2, 3, 4, 5, 6],
                                    [0, 1, 2, 3, 4, 5, 6],
                                    [0, 1, 2, 3, 4, 5, 6],
                                    [7, 8, 9, 10, 11, 12, 13],
                                    [7, 8, 9, 10, 11, 12, 13],
                                    [7, 8, 9, 10, 11, 12, 13],
                                    [14, 15, 16, 17, 18, 19, 20],
                                    [14, 15, 16, 17, 18, 19, 20],
                                    [14, 15, 16, 17, 18, 19, 20]],
                                    columns=list('ABCDEFG'))

        self.ratio_test_data = pd.DataFrame([
                                    [0, 1, 2, 3, 4, 5, 6],
                                    [7, 8, 9, 10, 11, 12, 13],
                                    [14, 15, 16, 17, 18, 19, 20]],
                                    columns=list('ABCDEFG'))

    def test_trials(self):
        """
        Test 'trials' setting.
        """
        df_splitter = dataframe_splitter.DataFrameSplitter(method="trials")
        train_data, test_data = df_splitter.train_test_spilt(self.dataset, 
                                                            ["E", "F"],  
                                                            trial_col="F", 
                                                            train_trials=[12,19],
                                                            verbose=0)
        train_data.reset_index(drop=True, inplace=True) 
        test_data.reset_index(drop=True, inplace=True)                                                                                                  
        assert_frame_equal(train_data, self.trial_train_data, check_dtype=False)
        assert_frame_equal(test_data, self.trial_test_data, check_dtype=False)
    
    def test_ratio(self):
        """
        Test 'ratio' setting.
        """
        """
        """
        df_splitter = dataframe_splitter.DataFrameSplitter(method="ratio")
        train_data, test_data = df_splitter.train_test_spilt(self.dataset, 
                                                            ["E", "F"],  
                                                            train_ratio=.80, 
                                                            verbose=0)
        train_data.reset_index(drop=True, inplace=True) 
        test_data.reset_index(drop=True, inplace=True)                                                                                                  
        assert_frame_equal(train_data, self.ratio_train_data, check_dtype=False)
        assert_frame_equal(test_data, self.ratio_test_data, check_dtype=False)

if __name__ == '__main__':
    unittest.main()