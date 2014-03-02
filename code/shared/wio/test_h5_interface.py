import unittest
import numpy as np
from h5_interface import * 


class TestH5Interface(unittest.TestCase):

    def test_basic_read_write(self):
        times = [1.2, 1.3, 1.4, 1.5]
        data = [2, 3, 4, 5]
        h5_file = 'test.h5'
        h5_path = 'test_A'
        write_h5_timeseries_base1(h5_file, h5_path, times, data)
        stimes, sdata = read_h5_timeseries_base1(h5_file, h5_path)
        for t1, t2 in zip(times, stimes):
            self.assertEqual(t1, t2)
        for d1, d2 in zip(data, sdata):
            self.assertEqual(t1, t2)

    def test_subgroup_read_write(self):
        times = [1.2, 1.3, 1.4, 1.5]
        data = [2, 3, 4, 5]
        h5_file = 'test.h5'
        h5_path = 'A/B/C'
        write_h5_timeseries_base1(h5_file, h5_path, times, data)
        stimes, sdata = read_h5_timeseries_base1(h5_file, h5_path)
        for t1, t2 in zip(times, stimes):
            self.assertEqual(t1, t2)
        for d1, d2 in zip(data, sdata):
            self.assertEqual(t1, t2)

    def test_repeated_read_write(self):
        h5_file = 'test.h5'
        h5_path = 'test_B'
        for i in range(5):
            times = np.arange(1.1, 1.5, 0.1)
            data = np.random.randn(4)       
            write_h5_timeseries_base1(h5_file, h5_path, times, data)
            stimes, sdata = read_h5_timeseries_base1(h5_file, h5_path)
            for t1, t2 in zip(times, stimes):
                self.assertEqual(t1, t2)
            for d1, d2 in zip(data, sdata):
                self.assertEqual(t1, t2)
            
    def test_file_not_present(self):
        h5_file = 'test2.h5'
        h5_path = 'test_A'
        times = [1.2, 1.3, 1.4, 1.5]
        data = [2, 3, 4, 5]
        sol_times, sol_data = [], []
        stimes, sdata = read_h5_timeseries_base1(h5_file, h5_path)
        for t1, t2 in zip(sol_times, stimes):
            self.assertEqual(t1, t2)
        for d1, d2 in zip(sol_data, sdata):
            self.assertEqual(t1, t2)

    def test_file_not_present(self):
        h5_file = 'test3.h5'
        h5_path = 'first_path'
        times = [1.2, 1.3, 1.4, 1.5]
        data = [2, 3, 4, 5]
        write_h5_timeseries_base1(h5_file, h5_path, times, data)
        h5_path = 'some_other_path'
        sol_times, sol_data = [], []
        stimes, sdata = read_h5_timeseries_base1(h5_file, h5_path)
        for t1, t2 in zip(sol_times, stimes):
            self.assertEqual(t1, t2)
        for d1, d2 in zip(sol_data, sdata):
            self.assertEqual(t1, t2)
            
                
if __name__ == '__main__':
    unittest.main()
