/* Daniel B. Stouffer daniel.stouffer@canterbury.ac.nz */
/* Originally written: February 2005 */
/* Updated: January 2014 */

#include <vector>

#include "data_tools.hpp"
#include "segmentation_tools.hpp"

using namespace std;

bool SegmentationFukuda(vector<double> x,vector<double> y,vector<double> integral,vector<int> starts,int segment,int& cutpoint);
void SegmentDataFukuda(vector<double> x,vector<double> y,vector<double> integral,vector<int>& starts);
