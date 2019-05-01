//
//  fast_ALS.cpp
//  fast_ALS
//
//  Created by yangsong on 5/1/19.
//  Copyright © 2019 yangsong. All rights reserved.
//

#include "fast_ALS.hpp"


const int userCount = 10000;
const int itemCount = 100000;

class MF_fastALS{
    int factors = 10;
    int maxIter = 500;     // maximum iterations.
    double reg = 0.01;     // regularization parameters
    double w0 = 1;
    double init_mean = 0;  // Gaussian mean for init V
    double init_stdev = 0.01; // Gaussian std-dev for init V
    
}
