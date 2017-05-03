#include <stdio.h>
#include <iostream>
#include "include/feature_matching.h"

// Define 'USE_SURF' to use USE_SURF keypoints instead of KAZE for comparation
#define USE_SURF
#define USE_SIFT 0

#define USE_XFEATURES2D  USE_SURF || USE_SIFT

#ifdef USE_XFEATURES2D
//#include "opencv2/xfeatures2d/nonfree.hpp"
#endif

void readme();
using namespace std;
using namespace FEATURE_MATCHING_NS;

int main(int argc, char** argv)
{
  if( argc != 3 )
    {
      std::cout <<  "Argument number: " << argc << std::endl;
      for (int i =0 ; i < argc; ++i)
        std::cout << string(argv[i]);
      std::cout << std::endl;
      readme();
      return -1;
    }

  bool jsonFlag = true;
  FeatureMatchingOptions opts;
  FeatureMatching featureMatching(argv[1], argv[2], opts, jsonFlag);

  if (featureMatching.process()) {
      cout << "Feature matching success!" << endl;
    }
  else {
      cout << "Feature matching FAIL!!!" << endl;
    }

  return 0;
}

/** @function readme */
void readme()
{ std::cout << " Usage: ./feature_matching <img1> <img2>" << std::endl; }

