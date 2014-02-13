'''
Author: Peter
Description:

'''
# standard imports
import unittest

# nonstandard imports
import manage_parts
from mongo_retrieve import mongo_query
import total_size

class TestPartManagement(unittest.TestCase):

    def setUp(self):
        '''
        This test set uses the first encoded_outline document in the database that it finds.
        '''
        self.big_entry = mongo_query({'data_type': 'encoded_outline'}, find_one=True)
        assert len(self.big_entry['data']) > 0
        assert isinstance(self.big_entry['data'], dict)

        self.number_of_parts = 7

    def test_timedict_split_and_combine(self, verbose=False):
        ''' This test takes the timedict from the test entry splits it into parts and recombines them
        '''
        test_timedict = self.big_entry.get('data', {})
        timedict_parts, part_indicies = manage_parts.split_timedict_into_parts(test_timedict,
                                                                               number_of_parts=self.number_of_parts)
        if verbose:
            print len(timedict_parts), part_indicies
            for i in timedict_parts:
                print len(i)

        recombined_timedict = manage_parts.combine_timedict_parts(timedict_parts, part_indicies)
        # recombined and origional timedicts must have same amount of timepoints
        self.assertEqual(len(test_timedict), len(recombined_timedict))

        # values for recombined dict must be identical to origional values
        for t in test_timedict:
            self.assertEqual(test_timedict[t], recombined_timedict[t])


    def test_entry_split_and_combine(self, verbose=False):
        """
        Repeatedly splits and combines entries to see if they stay consistent.
        Specifically: splits the big entry, recombines it, and splits it again.
        Tests if original entry and recombined entry are the same.
        Tests if first set and second set of split entries are the same.
        """
        origional_entry = self.big_entry.copy()
        del origional_entry['part']
        entries = manage_parts.split_entry_into_part_entries(origional_entry, num_parts=self.number_of_parts)
        self.assertEqual(len(entries), self.number_of_parts)

        reconstructed_entry = manage_parts.combine_part_entries_to_entry(entries)
        print reconstructed_entry['part'], self.big_entry['part']
        self.assertTrue('data' in reconstructed_entry)
        self.assertTrue('part' in reconstructed_entry)
        for atr in self.big_entry:
            if atr not in ['data', 'part', '_id']:
                self.assertEqual(self.big_entry[atr], reconstructed_entry[atr]), '{} not equal'.format(atr)
            else:
                'data gets special treatment'
                origional_timedict = self.big_entry['data']
                recombined_timedict = reconstructed_entry['data']
                self.assertEqual(len(origional_timedict), len(recombined_timedict))
                for t in origional_timedict:
                    self.assertEqual(origional_timedict[t], recombined_timedict[t])

        resplit_entires = manage_parts.split_entry_into_part_entries(reconstructed_entry, num_parts=self.number_of_parts)
        for e1, e2 in zip(entries, resplit_entires):
            for i in e1:
                self.assertEqual(e1[i], e2[i], '{i} not equal'.format(i=i))


    def test_split_number(self, verbose=True):
        ''' This test sees if the timedict will be split into the correct number of parts by changing the size limits.
        '''
        test_timedict = self.big_entry.get('data', {})
        actual_size = total_size.total_size(test_timedict)
        if verbose:
            print 'timedict size =', actual_size


        def split_timedict_by_N(N):

            max_size = actual_size / N + 1
            timedict_parts, part_indicies = manage_parts.split_timedict_into_parts(test_timedict,
                                                                                   max_size_of_timedict=max_size)
            return part_indicies

        for N in [1, 2, 3, 7, 12]:
            part_indicies = split_timedict_by_N(N=N)
            self.assertEqual(N, len(part_indicies))
            test_part_indicies = set(['{0}of{1}'.format(i, N) for i in range(1,N+1)])
            # make sure part indicies and the test_part_indicies contain same set of elements
            self.assertEqual(N, len(set(part_indicies) & test_part_indicies))


    # Todo: write a test that actually writes/pulls entries from database


if __name__ == '__main__':
    unittest.main()
