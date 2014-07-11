/* dbs */

#include <vector>

using namespace std;

void FindANCOVAStats(vector<double> x,vector<double> y,int start,int finish,double& SS_x,double& SS_y,double& SC);

void FindANCOVA(vector<double> x,vector<double> y,vector<int> starts,int segment,int max,double& F,double& p_F);

bool SegmentationFukudaAndANCOVA(vector<double> x,vector<double> y,vector<double> integral,vector<int> starts,int segment,int& cutpoint);

void SegmentDataANCOVA(vector<double> x,vector<double> y,vector<double> integral,vector<int>& starts);