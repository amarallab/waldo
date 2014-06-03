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

using namespace std;

void SortData(vector<double>& x,vector<double>& y){
  int i,j,min;
  double temp;

  for(i=0;i<x.size();++i){
    min=i;
    for(j=i+1;j<x.size();++j)
      if(x[j]<x[i])
	min=j;
    
    temp=x[i];
    x[i]=x[min];
    x[min]=temp;

    temp=y[i];
    y[i]=y[min];
    y[min]=temp;
  }
}

void SortData(vector<double>& x){
  int i,j,min;
  double temp;

  for(i=0;i<x.size();++i){
    min=i;
    for(j=i+1;j<x.size();++j)
      if(x[j]<x[i])
	min=j;
    
    temp=x[i];
    x[i]=x[min];
    x[min]=temp;
  }
}

double normaldistributionpdf(double x,double m,double s){
  double pi = acos(-1.0);
  double y = 1.0/s/sqrt(2*pi)*exp(-(x-m)*(x-m)/2/s/s);
  return y;
}

double normaldistributioncdf(double x,double m,double s){
  double pi = acos(-1.0);
  double y = 0.5*(1+erf((x-m)/s/sqrt(2.0)));
  return y;
}

double probks(double alam){
  double EPS1=0.001;
  double EPS2=1.0e-8;
  double a2,fac=2.0,sum=0.0,term,termbf=0.0;

  a2 = -2.0*alam*alam;
  for(int j=1;j<=100;j++){
    term=fac*exp(a2*j*j);
    sum += term;
    if (fabs(term) <= EPS1*termbf || fabs(term) <= EPS2*sum)
      return sum;
    fac = -fac;
    termbf=fabs(term);
  }
  return 1.0;
}

void kstest(vector<double> data, double m, double s, double (*func)(double,double,double), double& d, double& prob){
  double probks(double alam);
  //void sort(unsigned long n, float arr[]);
  double dt,en,ff,fn,fo=0.0;

  sort(data.begin(),data.end());
  en=data.size();
  d=0.0;
  for(int j=1;j<=data.size();j++){
    fn=j/en;
    ff=(*func)(data[j-1],m,s);
    if(fabs(fo-ff) > fabs(fn-ff))
      dt=fabs(fo-ff);
    else
      dt=fabs(fn-ff);
    if(dt > d)
      d=dt;
    fo=fn;
  }
  en=sqrt(en);
  prob=probks((en+0.12+0.11/en)*(d));
}

void linearregression(char *filename,double& m,double& b,double& Rsq){

  ifstream infile;
  infile.open(filename);

  double p,q;
  vector<double> x,y;

  while(infile >> p >> q){
    x.push_back(p);
    y.push_back(q);
  }

  int i,size = x.size();

  double sx=0,sy=0,sxsq=0,sysq=0,sxy=0;

  for(i=0;i<size;++i){
    sx += x[i];
    sxsq += x[i] * x[i];
    sy += y[i];
    sysq += y[i] * y[i];
    sxy += x[i] * y[i];
  }
  m = (size*sxy-sx*sy)/double(size*sxsq-sx*sx);
  b = (sy-m*sx)/double(size);

  Rsq=0;
  for(i=0;i<size;++i)
    Rsq += (y[i]-(m*x[i]+b))*(y[i]-(m*x[i]+b));

}

void linearregression(const vector<double>& x,const vector<double>& y,double& m,double& b,double& Rsq){
  if(x.size()!=y.size()){
    cout << "Vectors not the same size!\n";
    exit(0);
  }

  int i,size = x.size();
  double sx=0,sy=0,sxsq=0,sysq=0,sxy=0;

  for(i=0;i<size;++i){
    sx += x[i];
    sxsq += x[i] * x[i];
    sy += y[i];
    sysq += y[i] * y[i];
    sxy += x[i] * y[i];
  }
  m = (size*sxy-sx*sy)/double(size*sxsq-sx*sx);
  b = (sy-m*sx)/double(size);

  Rsq=0;
  for(i=0;i<size;++i)
    Rsq += (y[i]-(m*x[i]+b))*(y[i]-(m*x[i]+b));
}

void linearregression(const vector<double>& x,const vector<double>& y,int start,int finish,double& m,double& b,double& Rsq){
  if(x.size()!=y.size()){
    cout << "Vectors not the same size!\n";
    exit(0);
  }

  if(start>finish){
    cout << "You cannot start before you finish!\n";
    exit(0);
  }

  int i,size=finish+1-start;
  double sx=0,sy=0,sxsq=0,sysq=0,sxy=0;

  for(i=start;i<=finish;++i){
    sx += x[i];
    sxsq += x[i] * x[i];
    sy += y[i];
    sysq += y[i] * y[i];
    sxy += x[i] * y[i];
  }
  m = (size*sxy-sx*sy)/double(size*sxsq-sx*sx);
  b = (sy-m*sx)/double(size);

  Rsq=0;
  for(i=start;i<=finish;++i)
    Rsq += (y[i]-(m*x[i]+b))*(y[i]-(m*x[i]+b));
}

void loglinearregression(char *filename,double& m,double& b,double& Rsq){

  ifstream infile;
  infile.open(filename);

  double p,q;
  vector<double> x,y;

  while(infile >> p >> q){
    x.push_back(p);
    y.push_back(q);
  }

  int i,size = x.size();

  double sx=0,sy=0,sxsq=0,sysq=0,sxy=0;

  for(i=0;i<size;++i){
    sx += x[i];
    sxsq += x[i]*x[i];
    sy += log((y[i]));
    sysq += log(y[i])*log(y[i]);
    sxy += x[i]*log(y[i]);
  }
  m = (size*sxy-sx*sy)/double(size*sxsq-sx*sx);
  b = exp((sy-m*sx)/double(size));

  Rsq=0;
  for(i=0;i<size;++i)
    Rsq += (y[i]-(b*exp(m*x[i])))*(y[i]-(b*exp(m*x[i])));
}

void loglogregression(char *filename,double& m,double& b,double& Rsq){

  ifstream infile;
  infile.open(filename);

  double p,q;
  vector<double> x,y;

  while(infile >> p >> q){
    x.push_back(p);
    y.push_back(q);
  }

  int i,size = x.size();

  double sx=0,sy=0,sxsq=0,sysq=0,sxy=0;

  for(i=0;i<size;++i){
    sx += log(x[i]);
    sxsq += log(x[i])*log(x[i]);
    sy += log(y[i]);
    sysq += log(y[i])*log(y[i]);
    sxy += log(x[i])*log(y[i]);
  }
  m = (size*sxy-sx*sy)/double(size*sxsq-sx*sx);
  b = exp((sy-m*sx)/double(size));

  Rsq=0;
  for(i=0;i<size;++i)
    Rsq += (y[i]-(b*pow(x[i],m)))*(y[i]-(b*pow(x[i],m)));
}

double gammln(double xx){
  int j;
  double x,y,tmp,ser;
  static const double cof[6] ={76.18009172947146,-86.50532032941677,24.01409824083091,-1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5};
  static const double stp=2.5066282746310005;
  
  y=x=xx;
  tmp=x+5.5;
  tmp = (x+0.5)*log(tmp) -tmp;
  ser=1.000000000190015;

  for(j=0;j<6;++j){
    y+=1;
    ser+=cof[j]/double(y);
  }
  return(tmp+log(stp*ser/double(x)));    
}

double gcf(double a,double x){
  int itmax=100000;
  double eps=3E-7,fpmin=1E-30;
  
  int i;
  double an,b,c,d,del,h,gln;

  gln=gammln(a);
  b=x+1.0-a;
  c=1.0/fpmin;
  d=1.0/b;
  h=d;
  for(i=1;i<=itmax;++i){
    an = -i*(i-a);
    b=b+2.0;
    d=an*d+b;
    if(fabs(d) < fpmin)
      d=fpmin;
    c=b+an/c;
    if(fabs(c) < fpmin)
      c=fpmin;
    d=1.0/d;
    del=d*c;
    h=h*del;
    if(fabs(del-1.0) < eps)
      return(exp(-x+a*log(x)-gln)*h);
  }
  if(i>itmax){
    cout << "a too large, itmax too small in gcf\n";
    exit(0);
  }
}

double gser(double a,double x){
  int n;
  double ap,del,sum,gln;
  int itmax=100000;
  double eps=3E-7;
  
  gln=gammln(a);
  if(x <= 0){
    if(x < 0){
      cout << "x < 0 in gser\n";
      exit(0);
    }
    return(0);
  }
  else{
    ap=a;
    sum=1.0/a;
    del=sum;
    for(n=1;n<=itmax;++n){
      ap=ap+1;
      del=del*x/ap;
      sum=sum+del;
      if(fabs(del) < fabs(sum)*eps)
	return(sum*exp(-x+a*log(x)-gln));
    }
    if(n>itmax){
      cout << "a too large, itmax too small in gser\n";
      exit(0);
    }
  }
}

double gammq(double a,double x){
  double gammq,gammcf,gamser,gln;
  if(x < 0 || a <= 0){
    cout << "Bad arguments in gammq.\n";
    exit(0);
  }
  else{
    if(x < a+1.0)
      return(1.0-gser(a,x));
    else
      return(gcf(a,x));
  }
}

double gammp(double a,double x){
  double gammp,gammcf,gamser,gln;
  if(x < 0 || a <= 0){
    cout << "Bad arguments in gammp.\n";
    exit(0);
  }
  else{
    if(x < a+1.0){
      return(gser(a,x));
    }
    else{
      return(1.0-gcf(a,x));
    }
  }
}

// double erf(double x){
//   if(x<0)
//     return(-gammp(0.5,x*x));
//   else
//     return(gammp(0.5,x*x));	   
// }

double betacf(double a,double b,double x){
  int maxit=1000;
  double eps=numeric_limits<double>::epsilon();
  double fpmin=numeric_limits<double>::min()/double(eps);
  int m,m2;
  double aa,c,d,del,h,qab,qam,qap;

  qab=a+b;
  qap=a+1.0;
  qam=a-1.0;
  c=1.0;
  d=1.0-qab*x/double(qap);
  if(fabs(d) < fpmin)
    d=fpmin;
  d=1.0/double(d);
  h=d;

  for(m=1;m<=maxit;m++){
    m2=2*m;
    aa=m*(b-m)*x/double((qam+m2)*(a+m2));
    d=1.0+aa*d;
    if(fabs(d) < fpmin)
      d=fpmin;
    c=1.0+aa/double(c);
    if(fabs(c) < fpmin)
      c=fpmin;
    d=1.0/double(d);
    h*=d*c;
    aa=-(a+m)*(qab+m)*x/double((a+m2)*(qap+m2));
    d=1.0+aa*d;
    if(fabs(d) < fpmin)
      d=fpmin;
    c=1.0+aa/double(c);
    if(fabs(c) < fpmin)
      c=fpmin;
    d=1.0/double(d);
    del=d*c;
    h*=del;
    if(fabs(del-1.0) <= eps)
      break;
  }
  if(m > maxit){
    cout << "a or b too big, or maxit too small in betacf\n";
    exit(0);
  }
  return(h);
}

double betai(double a,double b,double x){
  double bt;

  if(x<0.0 || x>1.0){
    cout << "Bad x in routine betai\n";
    exit(0);
  }
  if(x==0.0 || x==1.0)
    bt=0.0;
  else
    bt=exp(gammln(a+b)-gammln(a)-gammln(b)+a*log(x)+b*log(1.0-x));
  
  if(x<(a+1.0)/double(a+b+2.0))
    return (bt*betacf(a,b,x)/double(a));
  else
    return (1.0-bt*betacf(b,a,1.0-x)/double(b));
}

void mean_stdev(vector<double> data,double& mean,double& stdev){
  double sx,sxsq;
  sx=sxsq=0;
  for(int i=0;i<data.size();++i){
    sx+=data[i];
    sxsq+=data[i]*data[i];
  }
  sx /= (data.size());
  sxsq /= (data.size());

  mean = sx;
  stdev = sqrt(sxsq-sx*sx);
}

void mean_stdev(vector<double> data,double& mean,double& stdev,int start,int finish){
  double sx,sxsq;
  sx=sxsq=0;
  for(int i=start;i<=finish;++i){
    sx+=data[i];
    sxsq+=data[i]*data[i];
  }
  sx /= (finish-start+1);
  sxsq /= (finish-start+1);

  mean = sx;
  stdev = sqrt(sxsq-sx*sx);
}

double PooledVariance(int N1,int N2,double s1,double s2){
  return (sqrt(((N1-1)*s1*s1 + (N2-1)*s2*s2)/double(N1+N2-2)) * sqrt(1.0/N1+1.0/N2));
}

double ProbTMax(double t_max,double N){
  double eta = 4.19*log(N)-11.54;
  double delta = 0.40;
  double nu = N-2.0;
  return (1-pow(1-betai(delta*nu,delta,nu/double(nu+t_max*t_max)),eta));
}

int HammingDistanceFromTarget(vector<int> idealgroups,vector<int> groups){
  if(idealgroups.size()!=groups.size()){
    cout << "Invalid option in 'HammingDistanceFromTarget'.\n";
    exit(0);
  }
  int count=0;
  for(int i=0;i<idealgroups.size();++i)
    if(idealgroups[i]!=groups[i])
      ++count;
  return(count);
}
