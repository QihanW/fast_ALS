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
public:
    int factors = 10;
    int maxIter = 500;     // maximum iterations.
    double reg = 0.01;     // regularization parameters
    double w0 = 1;
    double init_mean = 0;  // Gaussian mean for init V
    double init_stdev = 0.01;
    SpMat trainMatrix;// Gaussian std-dev for init V
    MatrixXd U;
    MatrixXd V;
    MatrixXd SU;
    MatrixXd SV;
    std::vector<double>  prediction_users, prediction_items;
    std::vector<double> rating_users, rating_items;
    std::vector<double> w_users, w_items;
    
    bool showprogress;
    bool showloss;
    SpMat W;
    std::vector<double> Wi;
    double w_new = 1;
    MF_fastALS(SpMat trainMatrix, std::vector<rating> testRatings,
               int topK, int threadNum, int factors, int maxIter, double w0, double alpha, double reg, double init_mean, double init_stdev, bool showprogress, bool showloss);
    double predict (int u, int i);
    double showLoss (int iter, long start, double loss_pre);
    double loss();

private:
    void initS();

protected:
    void buildModel();
    void update_user(int u);
    void update_item(int i);
};




//类构造函数
MF_fastALS::MF_fastALS(SpMat trainMatrix, std::vector<rating> testRatings,
               int topK, int threadNum, int factors, int maxIter, double w0, double alpha, double reg, double init_mean, double init_stdev, bool showprogress, bool showloss){
    //trainMatrix是列压缩的
    trainMatrix = trainMatrix;
    testRatings = testRatings;
    topK = topK;
    threadNum = threadNum;
    factors = factors;
    maxIter = maxIter;
    w0 = w0;
    reg = reg;
    init_mean = init_mean;
    init_stdev = init_stdev;
    showloss = showloss;
    showprogress = showprogress;
    
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
    double Wi[itemCount];
    for (int i = 0; i < itemCount; i ++)
        Wi[i] = w0 * p[i] / Z;
   
    SparseMatrix<double> W(userCount, itemCount);
    
    for (int u = 0; u < itemCount; u++)
        for( int i = trainMatrix.outerIndexPtr()[u]; i < trainMatrix.outerIndexPtr()[u+1]; i++)
            W.insert(u,trainMatrix.innerIndexPtr()[i]) = 1;
    //在栈里不需要释放内存m，以后需要改成new分配的！！
    double prediction_users [userCount];
    double prediction_items [itemCount];
    double rating_users [userCount];
    double rating_items [itemCount];
    double w_users [userCount];
    double w_items [itemCount];
    
    //Init model parameters
    MatrixXd U(userCount, factors);
    MatrixXd V(userCount, factors);
    
    //高斯分布 初始化稠密矩阵 基于给定的均值标准差
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
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
    SU = U.transpose()*U;
    MatrixXd SV(factors, factors);
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
        if (showprogress)
            showProgress(iter, start, testRatings);
        // Show loss
        if (showloss)
            loss_pre = showLoss(iter, start, loss_pre);
        
    } // end for iter
}

void MF_fastALS::update_user(int u) {
    //现在需要把col-major的trainMatrix改成row-major的, how??
    vector<int> itemList;
    for (int i = trainMatrix.outerIndexPtr()[u]; i < trainMatrix.outerIndexPtr()[u+1] ; i++)
        itemList.push_back(trainMatrix.innerIndexPtr()[i]);
    if (itemList.size() == 0)        return;    // user has no ratings
    // prediction cache for the user
    
    for (int i : itemList) {
        prediction_items[i] = predict(u, i);
        rating_items[i] = trainMatrix.coeffRef(u, i);
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
    vector<int> userList;
    for (int j = trainMatrix.outerIndexPtr()[i]; i < trainMatrix.outerIndexPtr()[i+1] ; j++)
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
    return U.row(u)*V.col(i);
}
double MF_fastALS::showLoss(int iter, long start, double loss_pre) {
    int64 start1 = LogTimeMM::getSystemTime();
    double loss_cur = loss();
    string symbol = loss_pre >= loss_cur ? "-" : "+";
    cout <<"Iter=" << iter << start1 - start << symbol << "loss:" <<loss_cur << LogTimeMM::getSystemTime() - start1 << endl;
   
    return loss_cur;
}
double MF_fastALS:: loss() {
    double L = reg * (U.squaredNorm() + V.squaredNorm());
    for (int u = 0; u < userCount; u++) {
        double l = 0;
        vector<int> itemList;
        for (int i = trainMatrix.outerIndexPtr()[u]; i < trainMatrix.outerIndexPtr()[u+1] ; i++)
            itemList.push_back(trainMatrix.innerIndexPtr()[i]);
        for (int i :itemList) {
            double pred = predict(u, i);
            l += W.coeffRef(u, i) * pow(trainMatrix.coeffRef(u, i) - pred, 2);
            l -= Wi[i] * pow(pred, 2);
        }
        l +=  U.row(u)*SV * U.row(u).transpose();
        L += l;
    }
    
    return L;
}





