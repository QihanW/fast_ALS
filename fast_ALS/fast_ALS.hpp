//
//  fast_ALS.cpp
//  fast_ALS
//
//  Created by yangsong on 5/1/19.
//  Copyright © 2019 yangsong. All rights reserved.
//

//#include "fast_ALS.hpp"

#ifndef FAST_ALS_H_
#define FAST_ALS_H_

#include <iostream>
#include <stdio.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <float.h>
#include <random>
#include <chrono>
#include <string>
#include "time.hpp"
#include "rating.hpp"
#include <math.h>


using namespace Eigen;

typedef Eigen::SparseMatrix<int> SpMat;
typedef Eigen::SparseMatrix<int, RowMajor> SpMat_R;
typedef Eigen::SparseMatrix<double> SpMat_D;
typedef Eigen::Triplet<double> T_d;
typedef Eigen::Matrix<double, Dynamic, Dynamic> MatrixXd;
typedef Eigen::Matrix<double, Dynamic, 1> VectorXd;

class MF_fastALS{
public:
    int factors;
    int maxIter;     // maximum iterations.
    double reg;     // regularization parameters
    double w0;
    double init_mean;  // Gaussian mean for init V
    double init_stdev;
    int itemCount;
    int userCount;
    int topK;
    double alpha;
    std::vector<Rating> testRatings;
    MatrixXd U;
    MatrixXd V;
    MatrixXd SU;
    MatrixXd SV;
    std::vector<double>  prediction_users, prediction_items;
    std::vector<double> rating_users, rating_items;
    std::vector<double> w_users, w_items;
    
    bool showprogress;
    bool showloss;
    SpMat trainMatrix;
    SpMat trainMatrix_R;
    SpMat_D W;
    std::vector<double> Wi;
    double w_new = 1;
    MF_fastALS(SpMat trainMatrix, SpMat_R trainMatrix_R, std::vector<Rating> testRatings,
               int topK, int factors, int maxIter, double w0, double alpha, double reg, double init_mean, double init_stdev, bool showprogress, bool showloss, int userCount, int itemCount);
    double predict (int u, int i);
    double showLoss (int iter, long start, double loss_pre);
    double loss();
    void buildModel();
    std::vector<double> evaluate_for_user(int u, int gtItem, int topK);
    double getHitRatio(std::vector<int> rankList, int gtItem);
    double getNDCG(std::vector<int> rankList, int gtItem);
    double getPrecision(std::vector<int> rankList, int gtItem);

private:
    void initS();

protected:
    void update_user(int u);
    void update_item(int i);
};




//类构造函数
MF_fastALS::MF_fastALS(
                       SpMat _trainMatrix,
                       SpMat_R _trainMatrix_R,
                       std::vector<Rating> _testRatings,
                       int _topK,
                       int _factors,
                       int _maxIter,
                       double _w0,
                       double _alpha,
                       double _reg,
                       double _init_mean,
                       double _init_stdev,
                       bool _showprogress,
                       bool _showloss,
                       int _userCount,
                       int _itemCount)
{
    //trainMatrix是列压缩的
    trainMatrix = _trainMatrix;
    trainMatrix_R = _trainMatrix_R;
    testRatings = _testRatings;
    topK = _topK;
    factors = _factors;
    maxIter = _maxIter;
    w0 = _w0;
    reg = _reg;
    alpha = _alpha;
    init_mean = _init_mean;
    init_stdev = _init_stdev;
    showloss = _showloss;
    showprogress = _showprogress;
    itemCount = _itemCount;
    userCount = _userCount;
    prediction_users.resize(userCount);
    prediction_items.resize(itemCount);
    rating_users.resize(userCount);
    rating_items.resize(itemCount);
    w_users.resize(userCount);
    w_items.resize(itemCount);
    
    
    double sum = 0, Z = 0;
  
    double p[itemCount];
    for (int i = 0; i < itemCount; i ++) {
        p[i] = trainMatrix.outerIndexPtr()[i+1] - trainMatrix.outerIndexPtr()[i];
        sum += p[i];
    }
    for ( int i = 0; i < itemCount; i++){
        p[i] /= sum;
        p[i] = pow(p[i], alpha);
        Z += p[i];
    }
    Wi.resize(itemCount);
    for (int i = 0; i < itemCount; i ++)
        Wi[i] = w0 * p[i] / Z;
    
    W.resize(userCount, itemCount);
    std::vector<T_d> tripletList;
    for (int u = 0; u < userCount; u++)
        for( int i = trainMatrix_R.outerIndexPtr()[u]; i < trainMatrix_R.outerIndexPtr()[u+1]; i++)
            tripletList.push_back(T_d(u,trainMatrix_R.innerIndexPtr()[i],1));
    W.setFromTriplets(tripletList.begin(), tripletList.end());
    
    //Init model parameters
    U.resize(userCount, factors);
    V.resize(itemCount, factors);
    
    //高斯分布 初始化稠密矩阵 基于给定的均值标准差
    unsigned seed = (unsigned) std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<double> distribution (init_mean, init_stdev);
    for ( int i = 0; i < userCount; i++)
        for ( int j = 0; j < factors; j++){
            U(i,j) = distribution (generator);
            V(i,j) = distribution (generator);}
    
    initS();
}


void MF_fastALS::initS(){
    //矩阵乘法
    SU = U.transpose()*U;//64*64
    SV.resize(factors, factors);
    for (int f = 0; f < factors; f ++) {
        for (int k = 0; k <= f; k ++) {
            double val = 0;
            for (int i = 0; i < itemCount; i ++)
                val += V(i, f) * V(i, k) * Wi[i];
            SV(f, k) = val;
            SV(k, f) = val;
        }
    }
}


void MF_fastALS::buildModel(){
    double loss_pre = DBL_MAX;
    for (int iter = 0; iter < maxIter; iter ++) {
        int64 start = LogTimeMM::getSystemTime();
        for (int u = 0; u < userCount; u ++) {
            update_user(u);
            
        }
        // Update item latent vectors
        for (int i = 0; i < itemCount; i ++) {
            update_item(i);
        }
        
        // Show progress
//        if (showprogress) {
////            showProgress(iter, start, testRatings);
//        }
        // Show loss
        if (showloss)
            loss_pre = showLoss(iter, start, loss_pre);
        
    } // end for iter
}

void MF_fastALS::update_user(int u) {
    //现在需要把col-major的trainMatrix改成row-major的, how??
    std::vector<int> itemList;
    for (int i = trainMatrix_R.outerIndexPtr()[u]; i < trainMatrix_R.outerIndexPtr()[u+1] ; i++)
        itemList.push_back(trainMatrix_R.innerIndexPtr()[i]);
    if (itemList.size() == 0)        return;    // user has no ratings
    // prediction cache for the user
    
    for (int i : itemList) {
        prediction_items[i] = predict(u, i);
        rating_items[i] = trainMatrix_R.coeffRef(u, i);
        w_items[i] = W.coeffRef(u,i);
    }
    
    //取dense矩阵的一行
    VectorXd oldVector = U.row(u);
    for (int f = 0; f < factors; f ++) {
        double numer = 0, denom = 0;
        // O(K) complexity for the negative part
        for (int k = 0; k < factors; k ++) {
            if (k != f)
                numer -= U(u, k) * SV(f, k);
        }
        //numer *= w0;
        // O(Nu) complexity for the positive part
        for (int i : itemList) {
            prediction_items[i] -= U(u, f) * V(i, f);
            numer +=  (w_items[i]*rating_items[i] - (w_items[i]-Wi[i]) * prediction_items[i]) * V(i, f);
            denom += (w_items[i]-Wi[i]) * V(i, f) * V(i, f);
        }
        denom += SV(f, f) + reg;
        
        // Parameter Update
        U(u,f) = numer / denom;
        
        // Update the prediction cache
        for (int i : itemList)
            prediction_items[i] += U(u, f) * V(i, f);
    } // end for f
    
    // Update the SU cache
    for (int f = 0; f < factors; f ++) {
        for (int k = 0; k <= f; k ++) {
            double val = SU(f, k) - oldVector[f] * oldVector[k]
            + U(u,f) * U(u,k);
            SU(f, k)= val;
            SU(k, f)= val;
        }
    }
}
void MF_fastALS::update_item(int i) {
    std::vector<int> userList;
    for (int j = trainMatrix.outerIndexPtr()[i]; j < trainMatrix.outerIndexPtr()[i+1] ; j++)
        userList.push_back(trainMatrix.innerIndexPtr()[j]);
    if (userList.size() == 0)        return; // item has no ratings.
    // prediction cache for the item
    for (int u : userList) {
        prediction_users[u] = predict(u, i);
        rating_users[u] = trainMatrix.coeffRef(u, i);
        w_users[u] = W.coeffRef(u, i);
    }
    
    
    VectorXd oldVector = V.row(i);
    for (int f = 0; f < factors; f++) {
        // O(K) complexity for the w0 part
        double numer = 0, denom = 0;
        for (int k = 0; k < factors;  k ++) {
            if (k != f)
                numer -= V(i, k) * SU(f, k);
        }
        numer *= Wi[i];
        
        // O(Ni) complexity for the positive ratings part
        for (int u : userList) {
            prediction_users[u] -= U(u, f) * V(i, f);
            numer += (w_users[u]*rating_users[u] - (w_users[u]-Wi[i]) * prediction_users[u]) * U(u, f);
            denom += (w_users[u]-Wi[i]) * U(u, f) * U(u, f);
        }
        denom += Wi[i] * SU(f, f) + reg;
        
        // Parameter update
        V(i, f) = numer / denom;
        // Update the prediction cache for the item
        for (int u : userList)
            prediction_users[u] += U(u, f) * V(i, f);
    } // end for f
    
    // Update the SV cache
    for (int f = 0; f < factors; f ++) {
        for (int k = 0; k <= f; k ++) {
            double val = SV(f, k) - oldVector[f] * oldVector(k) * Wi[i]
            + V(i, f) * V(i, k) * Wi[i];
            SV(f, k) = val;
            SV(k, f) = val;
        }
    }
}
double MF_fastALS::predict(int u, int i) {
    return U.row(u) * V.row(i).transpose();
}
double MF_fastALS::showLoss(int iter, long start, double loss_pre) {
    int64 start1 = LogTimeMM::getSystemTime();
    double loss_cur = loss();
    std::string symbol = loss_pre >= loss_cur ? "-" : "+";
    std::cout <<"Iter=" << iter << " " << start1 - start << symbol << " loss:" <<loss_cur <<" "<< LogTimeMM::getSystemTime() - start1 << std::endl;
   
    return loss_cur;
}
double MF_fastALS:: loss() {
    double L = reg * (U.squaredNorm() + V.squaredNorm());
    for (int u = 0; u < userCount; u++) {
        double l = 0;
        std::vector<int> itemList;
        for (int i = trainMatrix_R.outerIndexPtr()[u]; i < trainMatrix_R.outerIndexPtr()[u+1] ; i++)
            itemList.push_back(trainMatrix_R.innerIndexPtr()[i]);
        for (int i :itemList) {
            double pred = predict(u, i);
            l += W.coeffRef(u, i) * pow(trainMatrix_R.coeffRef(u, i) - pred, 2);
            l -= Wi[i] * pow(pred, 2);
        }
        l +=  U.row(u) * SV * U.row(u).transpose();
        L += l;
    }
    
    return L;
}
std::vector<double> MF_fastALS::evaluate_for_user(int u, int gtItem, int topK){
    std::vector<double> result(3);
    std::map<int, double> map_item_score;
    double maxScore;
//    int gtItem = testRatings[u].itemId;
//        double maxScore = predict(u, gtItem)
    maxScore = predict(u, gtItem);
    int countLarger = 0;
    for (int i = 0; i < itemCount; i++) {
        double score = predict(u, i);
        map_item_score.insert(std::make_pair(i,score));
        if (score >maxScore) countLarger++;
        if (countLarger > topK)  return result;
            //            if (countLarger > topK){
            //                hits[u]  = result[0];
            //                ndcgs[u] = result[1];
            //                precs[u] = result[2];
        }
    std::vector<int> rankList;
    std::vector<std::pair<int, double>>top_K(topK);
    std::partial_sort_copy( map_item_score.begin(),
                            map_item_score.end(),
                            top_K.begin(),
                            top_K.end(),
                            [](std::pair<const int, int> const& l,
                            std::pair<const int, int> const& r)
                            {
                                return l.second > r.second;
                            });
    for (auto const& p: top_K)
    {
        rankList.push_back(p.first);
        
    }
    result[0] = getHitRatio(rankList, gtItem);
    result[1] = getNDCG(rankList, gtItem);
    result[2] = getPrecision(rankList, gtItem);
    return result;
}
double MF_fastALS::getHitRatio(std::vector<int> rankList, int gtItem){
    for (int item : rankList) {
        if (item == gtItem)    return 1;
    }
    return 0;
}
double MF_fastALS::getNDCG(std::vector<int> rankList, int gtItem) {
    for (int i = 0; i < rankList.size(); i++) {
        int item = rankList[i];
        if (item == gtItem)
            return log(2) / log(i+2);
    }
    return 0;
}
double MF_fastALS::getPrecision(std::vector<int> rankList, int gtItem) {
    for (int i = 0; i < rankList.size(); i++) {
        int item = rankList[i];
        if (item == gtItem)
            return 1.0 / (i + 1);
    }
    return 0;
}
    




#endif

