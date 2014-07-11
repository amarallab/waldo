/* Daniel B. Stouffer daniel.stouffer@canterbury.ac.nz */
/* Originally written: February 2005 */
/* Updated: January 2014 */

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "data_tools.hpp"
#include "io_tools.hpp"
#include "ancova_tools.hpp"

using namespace std;

int main(int argc,char **argv){
  //  cout.setf(ios::fixed | ios::showpoint | ios::left);
  cout.precision(53);

  srand(time(NULL));

  vector<double> x,y;
  ReadDataFromStdin(x,y);

  x.insert(x.begin(),2*x[0]-x[1]);
  x.push_back(2*x[x.size()-1]-x[x.size()-2]);
  vector<double> derivative=y;
  vector<int> starts;

  vector<double> *interest;
  interest = &derivative;

  SegmentDataANCOVA(x,y,*interest,starts);
  x.erase(x.begin());
  x.pop_back();
  
  /*cout << x[0] << endl;
  for(int i=0;i<starts.size();++i)
    cout << x[starts[i]] << endl;
  cout << x[x.size()-1] << endl;*/
  
  PrintSegmentedData(x,y,starts);

  return(0);
}