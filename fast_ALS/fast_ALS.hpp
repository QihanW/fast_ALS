//
//  fast_ALS.hpp
//  fast_ALS
//
//  Created by yangsong on 5/1/19.
//  Copyright © 2019 yangsong. All rights reserved.
//
#include <iostream>
#include <stdio.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <float.h>
#include <random>
#include <chrono>
#include <string>
#include "time.cpp"
#include "rating.cpp"
#ifndef fast_ALS_hpp
#define fast_ALS_hpp

using namespace std;
using namespace Eigen;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;
typedef Eigen::Matrix<double, Dynamic, Dynamic> MatrixXd;
typedef Eigen::Matrix<double, Dynamic, 1> VectorXd;




#endif /* fast_ALS_hpp */
