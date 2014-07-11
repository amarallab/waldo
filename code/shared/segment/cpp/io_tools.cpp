/* Daniel B. Stouffer */

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

void ReadDataFromStdin(vector<double>& x,vector<double>& y){
  double p,q;
  x.clear();
  y.clear();
  while(cin >> p >> q){
    x.push_back(p);
    y.push_back(q);
  }

}

void ReadDataFromFile(char *filename,vector<double>& x,vector<double>& y){
  ifstream infile;
  double p,q;

  x.clear();
  y.clear();
  infile.open(filename);
  while(infile >> p >> q){
    x.push_back(p);
    y.push_back(q);
  }

}

void PrintData(vector<double> x,vector<double> y){
  for(int i=0;i<x.size();++i)
    cout << x[i] << " " << y[i] << endl;
}

void PrintLinearFit(vector<double> x,int start,int finish,double m,double b){
  for(int i=start;i<=finish;++i)
    cout << x[i] << " " << m*x[i]+b << endl;
  cout << endl;
}

void PrintLinearFit(vector<double> x,double m,double b){
  for(int i=0;i<x.size();++i)
    cout << x[i] << " " << m*x[i]+b << endl;
  cout << endl;
}

void PrintSegmentedData(vector<double> x,vector<double> y,vector<int> starts){
  int i,j,count=0;
  cout << x[count] << " " << y[count] << endl;
  ++count;
  for(i=0;i<starts.size()-1;++i){
    for(j=starts[i];j<starts[i+1];++j){
      cout << x[count] << " " << y[count] << endl;
      ++count;
    }
    cout << endl;
  }
  for(i=starts[i];i<x.size();++i){
    cout << x[count] << " " << y[count] << endl;
    ++count;
  }
}

void PrintSegmentedIntermediateData(vector<double> x,vector<double> data,vector<int> starts){
  int i,j;
  for(i=0;i<starts.size()-1;++i){
    for(j=starts[i];j<starts[i+1];++j)
      cout << x[j] << " " << data[j-1] << endl;
    cout << endl;
  }
  for(j=starts[i];j<=data.size();++j)
    cout << x[j] << " " << data[j-1] << endl;
}

void PrintIntermediateData(vector<double> x,vector<double> data){
  for(int i=0;i<data.size();++i)
    cout << x[i+1] << " " << data[i] << endl;
}

void PrintBestFitSlopes(vector<double> x,vector<double> y,vector<int> starts){
    double m,b,Rsq;
    int j,k;

    if(starts.size()==1){
      linearregression(x,y,0,x.size()-1,m,b,Rsq);
      for(j=0;j<x.size();++j)
	cout << x[j] << " " << m << endl;
    }else{
      linearregression(x,y,0,starts[1]-1,m,b,Rsq);
      //PrintLinearFit(x,starts[i],x.size()-1,m,b);
      for(j=0;j<starts[1];++j)
	cout << x[j] << " " << m << endl;
      
      for(k=1;k<starts.size()-1;++k){
	linearregression(x,y,starts[k],starts[k+1]-1,m,b,Rsq);
	//PrintLinearFit(x,starts[k],starts[k+1]-1,m,b);
	for(j=starts[k];j<starts[k+1];++j)
	  cout << x[j] << " " << m << endl;
      }

      linearregression(x,y,starts[k],x.size()-1,m,b,Rsq);
      //PrintLinearFit(x,starts[i],x.size()-1,m,b);
      for(j=starts[k];j<x.size();++j)
	cout << x[j] << " " << m << endl;
    }
}

void PrintBestFitLines(vector<double> x,vector<double> y,vector<int> starts){
    double m,b,Rsq;
    int j,k;

    if(starts.size()==1){
      linearregression(x,y,0,x.size()-1,m,b,Rsq);
      PrintLinearFit(x,0,x.size()-1,m,b);
    }else{
      linearregression(x,y,0,starts[1]-1,m,b,Rsq);
      PrintLinearFit(x,0,starts[1]-1,m,b);
      
      for(k=1;k<starts.size()-1;++k){
	linearregression(x,y,starts[k],starts[k+1]-1,m,b,Rsq);
	PrintLinearFit(x,starts[k],starts[k+1]-1,m,b);
      }

      linearregression(x,y,starts[k],x.size()-1,m,b,Rsq);
      PrintLinearFit(x,starts[k],x.size()-1,m,b);
    }
}

