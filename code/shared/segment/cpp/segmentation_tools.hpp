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

void SortData(vector<double>& x,vector<double>& y);

void SortData(vector<double>& x);

double normaldistributionpdf(double x,double m,double s);

double normaldistributioncdf(double x,double m,double s);

double probks(double alam);

void kstest(vector<double> data, double m, double s, double (*func)(double,double,double), double& d, double& prob);

void linearregression(char *filename,double& m,double& b,double& Rsq);

void linearregression(const vector<double>& x,const vector<double>& y,double& m,double& b,double& Rsq);

void linearregression(const vector<double>& x,const vector<double>& y,int start,int finish,double& m,double& b,double& Rsq);

void loglinearregression(char *filename,double& m,double& b,double& Rsq);

void loglogregression(char *filename,double& m,double& b,double& Rsq);

void ReadDataFromStdin(vector<double>& x,vector<double>& y);

void ReadDataFromFile(char *filename,vector<double>& x,vector<double>& y);

double gammln(double xx);

double gcf(double a,double x);

double gser(double a,double x);

double gammq(double a,double x);

double gammp(double a,double x);

double betacf(double a,double b,double x);

double betai(double a,double b,double x);

void mean_stdev(vector<double> data,double& mean,double& stdev);

void mean_stdev(vector<double> data,double& mean,double& stdev,int start,int finish);

double PooledVariance(int N1,int N2,double s1,double s2);

double ProbTMax(double t_max,double N);

void PrintData(vector<double> x,vector<double> y);

void PrintLinearFit(vector<double> x,int start,int finish,double m,double b);

void PrintLinearFit(vector<double> x,double m,double b);

void PrintSegmentedData(vector<double> x,vector<double> y,vector<int> starts);

void PrintSegmentedIntermediateData(vector<double> x,vector<double> data,vector<int> starts);

void PrintIntermediateData(vector<double> x,vector<double> data);

void PrintBestFitSlopes(vector<double> x,vector<double> y,vector<int> starts);

void PrintBestFitLines(vector<double> x,vector<double> y,vector<int> starts);

int HammingDistanceFromTarget(vector<int> idealgroups,vector<int> groups);