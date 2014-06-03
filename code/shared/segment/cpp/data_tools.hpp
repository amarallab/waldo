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

#include "segmentation_tools.hpp"
using namespace std;

void GenerateData(int argc,char **argv,vector<double>& x,vector<double>& y);

void GenerateData(int argc,char **argv,vector<double>& x,vector<double>& y,vector<int>& idealgroups);

void MakeDataNoisy(vector<double>& data,double s);

void MakeDataNoisy(vector<double>& x,vector<double>& y,double s_x,double s_y);

void GenerateNoisyData(int argc,char **argv,vector<double>& x,vector<double>& y);

void GenerateNoisyData(int argc,char **argv,vector<double>& x,vector<double>& y,vector<int>& idealgroups);

vector<double> Derivative(vector<double> x,vector<double> y);

vector<double> Curvature(vector<double> x,vector<double> y);

vector<double> AnalyticalCurvature(vector<double> x,vector<double> y);

vector<double> Integral(vector<double> x, vector<double> curvature);

// Scan across a data set and find the data point which, if cutting the data set there, would maximize the t test statistic
void FindIntegralSegmentation(vector<double> x,vector<double> y,vector<double> integral,vector<int> starts,int segment,double& t_max,int& max,double& p_tmax);

void CheckAdjacent(vector<double> x,vector<double> y,vector<double> integral,vector<int> starts,int segment,int max,double& p_left,double& p_right);

void CutData(vector<int>& starts,int segment,int cutpoint);

void ReJoinData(vector<int>& starts,int segment);
 