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

void GenerateData(int argc,char **argv,vector<double>& x,vector<double>& y){
  int i,j;

  int regimes=int(strtol(argv[1],NULL,10));
  
  vector<int> size;
  vector<double> alpha;

  int npoints=0;

  for(i=2;i<2+regimes;++i){
    size.push_back(int(strtol(argv[i],NULL,10)));
    alpha.push_back(strtod(argv[i+regimes],NULL));

    npoints+=size[i-2];
  }

  //cout << "there are " << npoints << " points\n";

  x.clear();
  y.clear();

  int k=0;
  double l=0;
  for(i=0;i<regimes;++i){
    for(j=0;j<size[i];++j){
      x.push_back(k*0.05);
      y.push_back(l);

      ++k;
      l+=alpha[i]*0.05;
    }
  }
}

void GenerateData(int argc,char **argv,vector<double>& x,vector<double>& y,vector<int>& idealgroups){
  int i,j;

  int regimes=int(strtol(argv[1],NULL,10));
  
  vector<int> size;
  vector<double> alpha;

  int npoints=0;

  for(i=2;i<2+regimes;++i){
    size.push_back(int(strtol(argv[i],NULL,10)));
    alpha.push_back(strtod(argv[i+regimes],NULL));

    npoints+=size[i-2];
  }

  //cout << "there are " << npoints << " points\n";

  x.clear();
  y.clear();
  idealgroups.clear();

  int k=0;
  double l=0;
  for(i=0;i<regimes;++i){
    for(j=0;j<size[i];++j){
      x.push_back(k*0.05);
      y.push_back(l);
      idealgroups.push_back(i);

      ++k;
      l+=alpha[i]*0.05;
    }
  }
}

void MakeDataNoisy(vector<double>& data,double s){
  double w,l,p;
  for(int i=0;i<data.size();++i){
    do{
      l = 2.0 * rand()/double(RAND_MAX) - 1.0;
      p = 2.0 * rand()/double(RAND_MAX) - 1.0;
      w = l * l + p * p;
    } while ( w >= 1.0 );
    
    w = sqrt( (-2.0 * log( w ) ) / w );

    data[i] += l*w*s;
  }
}

void MakeDataNoisy(vector<double>& x,vector<double>& y,double s_x,double s_y){
  double w,l,p;
  for(int i=0;i<x.size();++i){
    do{
      l = 2.0 * rand()/double(RAND_MAX) - 1.0;
      p = 2.0 * rand()/double(RAND_MAX) - 1.0;
      w = l * l + p * p;
    } while ( w >= 1.0 );
    
    w = sqrt( (-2.0 * log( w ) ) / w );
    x[i] += l*w*s_x;

    do{
      l = 2.0 * rand()/double(RAND_MAX) - 1.0;
      p = 2.0 * rand()/double(RAND_MAX) - 1.0;
      w = l * l + p * p;
    } while ( w >= 1.0 );
    
    w = sqrt( (-2.0 * log( w ) ) / w );
    y[i] += l*w*s_y;
  }
}

void GenerateNoisyData(int argc,char **argv,vector<double>& x,vector<double>& y){
  int i,j;

  int regimes=int(strtol(argv[1],NULL,10));
  
  vector<int> size;
  vector<double> alpha;

  int npoints=0;

  for(i=2;i<2+regimes;++i){
    size.push_back(int(strtol(argv[i],NULL,10)));
    alpha.push_back(strtod(argv[i+regimes],NULL));

    npoints+=size[i-2];
  }

  //  cout << "there are " << npoints << " points\n";

  x.clear();
  y.clear();

  int k=0;
  double l=0;
  for(i=0;i<regimes;++i){
    for(j=0;j<size[i];++j){
      x.push_back(k*0.05);
      y.push_back(l);

      ++k;
      l+=alpha[i]*0.05;
    }
  }

  if(argc>(1+regimes*2)+2)
    srand(int(strtol(argv[(1+regimes*2)+2],NULL,10)));
  
  double noise=strtod(argv[(1+regimes*2)+1],NULL);
  if(noise>0.0)
    MakeDataNoisy(y,noise);
}

void GenerateNoisyData(int argc,char **argv,vector<double>& x,vector<double>& y,vector<int>& idealgroups){
  int i,j;

  int regimes=int(strtol(argv[1],NULL,10));
  
  vector<int> size;
  vector<double> alpha;

  int npoints=0;

  for(i=2;i<2+regimes;++i){
    size.push_back(int(strtol(argv[i],NULL,10)));
    alpha.push_back(strtod(argv[i+regimes],NULL));

    npoints+=size[i-2];
  }

  //  cout << "there are " << npoints << " points\n";

  x.clear();
  y.clear();
  idealgroups.clear();

  int k=0;
  double l=0;
  for(i=0;i<regimes;++i){
    for(j=0;j<size[i];++j){
      x.push_back(k*0.05);
      y.push_back(l);
      idealgroups.push_back(i);

      ++k;
      l+=alpha[i]*0.05;
    }
  }

  if(argc>(1+regimes*2)+2)
    srand(int(strtol(argv[(1+regimes*2)+2],NULL,10)));
  
  double noise=strtod(argv[(1+regimes*2)+1],NULL);
  if(noise>0.0)
    MakeDataNoisy(y,noise);
}


/*void GenerateCurvedData(char **argv,vector<double>& x,vector<double>& y){
  int i,j;

  int regimes=int(strtol(argv[1],NULL,10));
  
  vector<int> size;
  vector<double> radius;

  int npoints=0;

  for(i=2;i<2+regimes;++i){
    size.push_back(int(strtol(argv[i],NULL,10)));
    alpha.push_back(strtod(argv[i+regimes],NULL));

    npoints+=size[i-2];
  }

  //  cout << "there are " << npoints << " points\n";

  x.clear();
  y.clear();

  int k=0;
  double l=0;
  for(i=0;i<regimes;++i){
    for(j=0;j<size[i];++j){
      x.push_back(k*0.05);
      y.push_back(l);

      ++k;
      l+=alpha[i]*0.05;
    }
  }

    /*
  for(i=0;i<n2;++i){
    x.push_back(4*(i+n1)/double(n1+n2));
    y.push_back(4*(a1*n1+i*a2)/double(n1+n2));
  }
}
*/

vector<double> Derivative(vector<double> x,vector<double> y){
  double y_prime;

  vector<double> derivative;

  int i;
  for(i=1;i<x.size()-1;++i){
    y_prime = (y[i+1]-y[i-1])/(x[i+1]-x[i-1]);
    derivative.push_back(y_prime);
  }

  return derivative;
}

vector<double> Curvature(vector<double> x,vector<double> y){
  double y_prime;
  double y_doubleprime;

  vector<double> curvature;

  int i;
  for(i=1;i<x.size()-1;++i){
    y_prime = (y[i+1]-y[i-1])/(x[i+1]-x[i-1]);
    y_doubleprime = (y[i+1]+y[i-1]-2.0*y[i])/((x[i+1]-x[i])*(x[i]-x[i-1]));
    
    curvature.push_back(y_doubleprime/pow(1+pow(y_prime,2),1.5));
  }

  return curvature;
}

vector<double> AnalyticalCurvature(vector<double> x,vector<double> y){
  double m1,m2,alpha,beta,c;
  double x0,y0,r;

  vector<double> curvature;

  int i;
  for(i=1;i<x.size()-1;++i){
    alpha = x[i]-x[i-1];
    beta = x[i+1]-x[i];
    m1 = (y[i]-y[i-1])/double(alpha);
    m2 = (y[i+1]-y[i])/double(beta);

    x0=(m2*alpha+m1*m1*m2*alpha+m1*beta+m1*m2*m2*beta)/2.0/(m1-m2);
    y0=-(alpha+m1*m1*alpha+beta+m2*m2*beta)/2.0/(m1-m2);

    c = -2.0*(m1-m2)/sqrt((1+m1*m1)*(1+m2*m2)*(alpha*alpha*(1+m1*m1)+2*alpha*(beta+beta*m1*m2)+beta*beta*(1+m2*m2)));
    
    curvature.push_back(c);
  }

  return curvature;
}

vector<double> Integral(vector<double> x, vector<double> curvature){
  vector<double> integral;

  int i;
  double j=0;

  integral.push_back(j);
  for(i=2;i<x.size()-1;++i){
    j += (x[i]-x[i-1])*(curvature[i-1]+curvature[i-2])/2.0;

    integral.push_back(j);
  }

  return integral;
}

// Scan across a data set and find the data point which, if cutting the data set there, would maximize the t test statistic
void FindIntegralSegmentation(vector<double> x,vector<double> y,vector<double> integral,vector<int> starts,int segment,double& t_max,int& max,double& p_tmax){
  int i,j,k;
  double m1,m2,s1,s2,SD,t;
  int begin,end;

  begin = starts[segment];
  if(segment==starts.size()-1)
    end = x.size()-2;
  else
    end = starts[segment+1]-1;

  // DEBUG: maybe minsize should be a parameter?
  int minsize=6;

  t_max=0;
  max = -1;

  for(i=begin+minsize-1;i<=end-minsize;++i){
    mean_stdev(integral,m1,s1,begin,i);
    mean_stdev(integral,m2,s2,i+1,end);

    SD = PooledVariance(i-begin+1,end-i,s1,s2);
    t = fabs((m1-m2)/SD);

    if(t>t_max){
      max = i;
      t_max = t;
    }
  }

  if(max == -1)
    p_tmax = 1;
  else
    p_tmax = ProbTMax(t_max,end-begin+1);
}

void CheckAdjacent(vector<double> x,vector<double> y,vector<double> integral,vector<int> starts,int segment,int max,double& p_left,double& p_right){
  int i,j,k;
  double m1,m2,s1,s2,SD,t;
  int begin,end;

  // DEBUG: maybe minsize should be a parameter?
  int minsize=6;

  if(segment>0){
    begin = starts[segment-1];
    end = max;

    mean_stdev(integral,m1,s1,begin,starts[segment]-1);
    mean_stdev(integral,m2,s2,starts[segment],end);

    SD = PooledVariance((starts[segment]-1)-begin+1,end-(starts[segment]-1),s1,s2);
    t = fabs((m1-m2)/SD);
    p_left = ProbTMax(t,end-begin+1);
  }else
    p_left = 0;

  if(segment<starts.size()-1){
    begin = max+1;
    if(segment==starts.size()-2)
      end = x.size()-2;
    else
      end = starts[segment+2]-1;

    mean_stdev(integral,m1,s1,begin,starts[segment+1]-1);
    mean_stdev(integral,m2,s2,starts[segment+1],end);

    SD = PooledVariance((starts[segment+1]-1)-begin+1,end-(starts[segment+1]-1),s1,s2);
    t = fabs((m1-m2)/SD);

    p_right = ProbTMax(t,end-begin+1);
  }else
    p_right = 0;
}

void CutData(vector<int>& starts,int segment,int cutpoint){
  starts.insert(starts.begin()+segment+1,cutpoint+1);
}

void ReJoinData(vector<int>& starts,int segment){
  starts.erase(starts.begin()+segment+1);
}

/*void CheckSegmentDistributions(vector<double> data,vector<int> starts,vector<double>& prob){
  vector<double> temp;
  int i,j,k;
  double m,s,d,p;

  double (*pfunc) (double,double,double);
  pfunc = &normaldistributioncdf;

  cout << "the party has begun\n";
  for(i=0;i<starts.size()-1;++i){
    temp.clear();
    for(j=starts[i];j<starts[i+1];++j)
      temp.push_back(data[j-1]);
    mean_stdev(temp,m,s);
    kstest(temp,m,s,pfunc,d,p);
    cout << i << " " << m << " " << s << " " << p << endl;
  }

  temp.clear();
  for(j=starts[i];j<data.size()+1;++j)
    temp.push_back(data[j-1]);
  mean_stdev(temp,m,s);
  kstest(temp,m,s,pfunc,d,p);
  cout << i << " " << m << " " << s << " " << p << endl;
 
}

void Analysis(int argc,char **argv,vector<double> x,vector<double> y){
  vector<double> curvature,integral,derivative,prob;
  vector<int> starts;
  int i,j,count;
  switch (argv[1][0])
    {
    case 'c':
      curvature = AnalyticalCurvature(x,y);
      // Print out the curvature data
      for(i=0;i<curvature.size();++i)
	cout << x[i+1] << " " << curvature[i] << endl;
      break;
    case 'i':
      curvature = AnalyticalCurvature(x,y);
      integral = Integral(x,curvature);
      if(argc>=3){
	if(argc==3 || argc==4){
	  switch (argv[2][0])
	    {
	    case 'f':
	      SegmentDataFukuda(x,y,integral,starts);
	      if(argc==4){
		if(argv[3][0]=='s'){
		  PrintSegmentedIntermediateData(x,integral,starts);
		}else{
		  cout << "Invalid option.\n";
		  exit(0);
		}
	      }else{
		PrintSegmentedData(x,y,starts);
	      }
	      break;
	    case 'A':
	      SegmentDataANCOVA(x,y,integral,starts);
	      if(argc==4){
		if(argv[3][0]=='s'){
		  PrintSegmentedIntermediateData(x,integral,starts);
		}else{
		  cout << "Invalid option.\n";
		  exit(0);
		}
	      }else{
		PrintSegmentedData(x,y,starts);
	      }
	      break;
	    default:
	      cout << "Invalid option.\n";
	      exit(0);
	    }
	}else{
	  cout << "Invalid option.\n";
	  exit(0);
	}
      }else{
	PrintIntermediateData(x,integral);
      }
      break;
    case 'd':
      derivative = Derivative(x,y);
      if(argc>=3){
	if(argc==3 || argc==4){
	  switch (argv[2][0])
	    {
	    case 'f':
	      SegmentDataFukuda(x,y,derivative,starts);
	      if(argc==4){
		if(argv[3][0]=='s'){
		  PrintSegmentedIntermediateData(x,derivative,starts);
		}else{
		  cout << "Invalid option.\n";
		  exit(0);
		}
	      }else{
		PrintSegmentedData(x,y,starts);
	      }
	      break;
	    case 'A':
	      SegmentDataANCOVA(x,y,derivative,starts);
	      if(argc==4){
		if(argv[3][0]=='s'){
		  PrintSegmentedIntermediateData(x,derivative,starts);
		}else{
		  cout << "Invalid option.\n";
		  exit(0);
		}
	      }else{
		//		PrintSegmentedData(x,y,starts);
		cout << "we ended up with " << starts.size() << " segments\n";
		CheckSegmentDistributions(derivative,starts,prob);
	      }
	      break;
	    default:
	      cout << "Invalid option.\n";
	      exit(0);
	    }
	}else{
	  cout << "Invalid option.\n";
	  exit(0);
	}
      }else{
	PrintIntermediateData(x,derivative);
      }
      break;
    default:
      cout << "That is not a valid option.\n";
      exit(0);
    }
}
*/
