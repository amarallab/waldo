
# standard imports
import unittest
import find_coils
import math

class TestCoilFunctions(unittest.TestCase):
    def setUp(self):
        self.acceptable_error = 10e-6

    def test_remove_loner_flags(self):

        def create_blank_flag_timedict(l=100):
            flag_timedict = {}
            for i in range(l): flag_timedict[str(i)+'?00'] = True
            return flag_timedict

        # all flags removed
        flag_timedict = create_blank_flag_timedict()
        flag_timedict['50?00'] = False
        new_flag_timedict = find_coils.remove_loner_flags(flag_timedict)
        self.assertEqual(create_blank_flag_timedict(), new_flag_timedict)
        flag_timedict['51?00'] = False
        new_flag_timedict = find_coils.remove_loner_flags(flag_timedict)
        self.assertEqual(create_blank_flag_timedict(), new_flag_timedict)
        # no flags removed
        flag_timedict['52?00'] = False
        new_flag_timedict = find_coils.remove_loner_flags(flag_timedict)
        self.assertEqual(flag_timedict, new_flag_timedict)
        flag_timedict['53?00'] = False
        new_flag_timedict = find_coils.remove_loner_flags(flag_timedict)
        self.assertEqual(flag_timedict, new_flag_timedict)
        flag_timedict['54?00'] = False
        new_flag_timedict = find_coils.remove_loner_flags(flag_timedict)
        self.assertEqual(flag_timedict, new_flag_timedict)
        flag_timedict['55?00'] = False
        new_flag_timedict = find_coils.remove_loner_flags(flag_timedict)
        self.assertEqual(flag_timedict, new_flag_timedict)


    '''
    def test_compute_basic_properties(self):
        self.assertEqual(propdict['duration'], 10.0)
        self.assertTrue(abs(propdict['bl_dist'] -  (math.sqrt(2*(100**2))/50.0) < acceptable_error))
    '''
        

if __name__ == '__main__':
    unittest.main()
