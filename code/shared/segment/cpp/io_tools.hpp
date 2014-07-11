/* Daniel B. Stouffer */

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>
#include <limits>
#include <map>

using namespace std;

void ReadDataFromStdin(vector<double>& x,vector<double>& y);
void ReadDataFromFile(char *filename,vector<double>& x,vector<double>& y);

void PrintData(vector<double> x,vector<double> y);

void PrintLinearFit(vector<double> x,int start,int finish,double m,double b);
void PrintLinearFit(vector<double> x,double m,double b);

void PrintSegmentedData(vector<double> x,vector<double> y,vector<int> starts);
void PrintSegmentedIntermediateData(vector<double> x,vector<double> data,vector<int> starts);

void PrintIntermediateData(vector<double> x,vector<double> data);

void PrintBestFitSlopes(vector<double> x,vector<double> y,vector<int> starts);

void PrintBestFitLines(vector<double> x,vector<double> y,vector<int> starts);
