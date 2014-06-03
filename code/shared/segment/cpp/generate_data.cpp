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
//#include "segmentation_tools.cpp"
using namespace std;

int main(int argc,char **argv){
  //  cout.setf(ios::fixed | ios::showpoint | ios::left);
  cout.precision(53);

  srand(time(NULL));

  vector<double> x,y,derivative;
  GenerateData(argc,argv,x,y);

  double noise_x = strtod(argv[argc-2],NULL);
  double noise_y = strtod(argv[argc-1],NULL);

  MakeDataNoisy(x,y,noise_x,noise_y);
  //PrintData(x,y);
  
  derivative=Derivative(x,y);
  PrintIntermediateData(x,derivative);

  return(0);
}
