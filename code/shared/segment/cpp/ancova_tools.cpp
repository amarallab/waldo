/* Daniel B. Stouffer daniel.stouffer@canterbury.ac.nz */
/* Originally written: February 2005 */
/* Updated: January 2014 */

/*#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
*/
#include <vector>

#include "data_tools.hpp"
#include "segmentation_tools.hpp"

using namespace std;

void FindANCOVAStats(vector<double> x,vector<double> y,int start,int finish,double& SS_x,double& SS_y,double& SC){
  double sx,sxsq,sy,sysq,sxy;
  double a,b;
 
  sx=sxsq=sy=sysq=sxy=0;
  for(int i=start;i<=finish;++i){
    sx+=x[i];
    sxsq+=x[i]*x[i];
    
    sy+=y[i];
    sysq+=y[i]*y[i];

    sxy+=x[i]*y[i];
  }

  SS_x = sxsq - sx*sx/double(finish-start+1);
  SS_y = sysq - sy*sy/double(finish-start+1);
  SC = sxy - sx*sy/double(finish-start+1);
}

void FindANCOVA(vector<double> x,vector<double> y,vector<int> starts,int segment,int max,double& F,double& p_F){
  double SS_ax,SS_ay,SC_a,SS_bx,SS_by,SC_b;
  double SS_wgx,SS_wgy,SC_wg,SS_effect,adjSS_wgy,SS_error;
  int N,ngroups,df_effect,df_error;
  int i,j,k;

  int begin,end;
  begin = starts[segment];

  if(segment==starts.size()-1)
    end = x.size()-2;
  else
    end = starts[segment+1]-1;

  FindANCOVAStats(x,y,begin,max,SS_ax,SS_ay,SC_a);
  FindANCOVAStats(x,y,max+1,end,SS_bx,SS_by,SC_b);

  N = end-begin+1;
  ngroups = 2;

  SS_wgx = SS_ax + SS_bx;
  SS_wgy = SS_ay + SS_by;
  SC_wg = SC_a + SC_b;

  SS_effect = SC_a*SC_a/SS_ax + SC_b*SC_b/SS_bx - SC_wg*SC_wg/SS_wgx;
  df_effect = ngroups-1;

  adjSS_wgy = SS_wgy - SC_wg*SC_wg/SS_wgx;
  SS_error = adjSS_wgy - SS_effect;

  df_error = (N-ngroups-1) - df_effect;

  F = (SS_effect/double(df_effect))/(SS_error/df_error);
  p_F = betai(df_error/2.0,df_effect/2.0,df_error/(df_error+df_effect*F));
}

bool SegmentationFukudaAndANCOVA(vector<double> x,vector<double> y,vector<double> integral,vector<int> starts,int segment,int& cutpoint){
  int i,j,k;
  int max;
  double t_max,p_tmax,F,p_F,p_left,p_right;

  FindIntegralSegmentation(x,y,integral,starts,segment,t_max,max,p_tmax);
  if(p_tmax!=1)
    CheckAdjacent(x,y,integral,starts,segment,max,p_left,p_right);
  else
    p_left = p_right = 1;

  if(p_tmax<0.05 && p_left<0.05 && p_right<0.05){
    FindANCOVA(x,y,starts,segment,max,F,p_F);
    if(p_F<0.05){
      cutpoint = max;
      return true;
    }
    else{
      cutpoint = -1;
      return false;
    }
  }
  else{
    cutpoint = -1;
    return false;
  }
}

void SegmentDataANCOVA(vector<double> x,vector<double> y,vector<double> integral,vector<int>& starts){
  bool cut=true;
  int i,j,k,cutpoint,count;

  starts.clear();
  starts.push_back(1);
  while(cut){
    cut = false;
    for(i=0;i<starts.size();++i){
      cut = SegmentationFukudaAndANCOVA(x,y,integral,starts,i,cutpoint);
      if(cut){
        CutData(starts,i,cutpoint);
        break;
      }
    }
  }
} 
