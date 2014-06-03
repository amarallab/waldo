/* Daniel B. Stouffer daniel.stouffer@canterbury.ac.nz */
/* Originally written: February 2005 */
/* Updated: January 2014 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <vector>

#include "data_tools.hpp"
#include "segmentation_tools.hpp"
using namespace std;

// Find the best place to cut a segmented time series based on the max t-test statistic so long as the difference is significant
bool SegmentationFukuda(vector<double> x,vector<double> y,vector<double> integral,vector<int> starts,int segment,int& cutpoint){
  int i,j,k;
  int max;
  double t_max,p_tmax,F,p_F,p_left,p_right;

  // At which data point is t greatest?
  FindIntegralSegmentation(x,y,integral,starts,segment,t_max,max,p_tmax);
  if(p_tmax!=1)
    // Compute the p-values associated with the the new segments and those to their left and right
    CheckAdjacent(x,y,integral,starts,segment,max,p_left,p_right);
  else
    p_left = p_right = 1;

  // If all three comparisons are significantly different, we can cut at the point corresponding to t_max 
  if(p_tmax<0.05 && p_left<0.05 && p_right<0.05){
    cutpoint = max;
    return true;
  }
  else{
    cutpoint = -1;
    return false;
  }
}

// Segment data as in Fukuda, Stanley, Amaral until there are no more "viable" cut points
void SegmentDataFukuda(vector<double> x,vector<double> y,vector<double> integral,vector<int>& starts){
  bool cut=true;
  int i,j,k,cutpoint,count;

  starts.clear();
  starts.push_back(1);
  while(cut){
    cut = false;
    for(i=0;i<starts.size();++i){
      cut = SegmentationFukuda(x,y,integral,starts,i,cutpoint);
      if(cut){
	      CutData(starts,i,cutpoint);
        break;
      }
    }
  }
}
