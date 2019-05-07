//
//  main.cpp
//  fast_ALS
//
//  Created by yangsong on 5/1/19.
//  Copyright © 2019 yangsong. All rights reserved.
//

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <assert.h>
#include "fast_ALS.hpp"
#include <stdio.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "time.hpp"
#include "rating.hpp"
#include <unordered_map>


using namespace Eigen;
typedef Eigen::SparseMatrix<int> SpMat;
typedef Eigen::SparseMatrix<int, RowMajor> SpMat_R;
typedef Eigen::Triplet<int> T;
typedef Eigen::Matrix<double, Dynamic, 1> VectorXd;


int main(int argc, const char * argv[])
{
    std::string dataset_name = "yelp";
    std::string method = "FastALS";
    double w0 = 10;
    bool showprogress = false;
    bool showloss = true;
    int factors = 64;
    int maxIter = 500;
    double reg = 0.01;
    double alpha = 0.75;
    double init_mean = 0;  // Gaussian mean for init V
    double init_stdev = 0.01;
    VectorXd hits;
    VectorXd ndcgs;
    VectorXd precs;
    
    
    
    //处理原始数据
    //为什么是static?
    static int topK = 10;
//    static int threadNum = 100;
    static int userCount;
    static int itemCount;
    
    
    std::cout <<"Holdone out splitting" <<std::endl;
    
    // Step 1. Construct data structure for sorting.
    std::cout << "Sort items for each user."<<std::endl;
    int64 startTime = LogTimeMM::getSystemTime();
   
    std::vector<std::vector<Rating>> user_ratings;
    char input_filename[] = "/Users/yangsong/Desktop/fast_ALS/data/yelp.rating";
    std::ifstream  fin(input_filename);
    if (!fin.is_open()) {
        fprintf(stderr, "Error: cannot open the file %s\n", input_filename);
        exit(EXIT_FAILURE);
    }
    std::string line;
    int user_id;
    int item_id;
    float score;
    long timestamp = 0;
    while (std::getline(fin, line)) {
        std::istringstream word(line);
        word >> user_id;
        word >> item_id;
        word >> score;
        word >> timestamp;
        
//        printf("uid: %d item_id: %d score: %f timestamp: %ld\n",
//                                                user_id,
//                                                item_id,
//                                                score,
//                                                timestamp);
        Rating rating(user_id,
                      item_id,
                      score,
                      timestamp);
//        printf("uid: %d item_id: %d score: %f timestamp: %ld\n",
//                                           rating.userId,
//                                           rating.itemId,
//                                           rating.score,
//                                           rating.timestamp);
        if (user_ratings.size() < rating.userId + 1){
            user_ratings.push_back(std::vector<Rating>());
        }
        user_ratings.rbegin() -> push_back(rating);
        userCount = fmax(userCount, rating.userId);
        itemCount = fmax(itemCount, rating.itemId);
    }
    userCount++;
    itemCount++;
    assert (userCount == user_ratings.size());
    //std::cout << userCount << "\r" << itemCount<< std::endl;
    
    
//    { // test
////        std::ofstream fout("/Users/yangsong/Desktop/fast_ALS/data/output.txt");
//        FILE *fout = fopen("/Users/yangsong/Desktop/fast_ALS/data/output.txt", "w");
//        for (const auto &user_vector : user_ratings) {
//            for (const auto & rating : user_vector) {
////                fout << rating.userId << "\t"
////                << rating.itemId << "\t"
////                << rating.score << "\t"
////                << rating.timestamp << std::endl;
//                fprintf(fout,"%d\t%d\t%.1f\t%ld\n",
//                                    rating.userId,
//                                    rating.itemId,
//                                    rating.score,
//                                    rating.timestamp);
//            }
//        }
//        fclose(fout);
//    }
    // Step 2. Sort the ratings of each user by time (small->large).
    
    for ( int u = 0; u < userCount; u++){
        sort(user_ratings[u].begin(),user_ratings[u].end(),LessSort);
    }
    
    std::cout << "sorting time:"<< LogTimeMM::getSystemTime() - startTime << std::endl;
    // Step 3. Generated splitted matrices (implicit 0/1 settings)
    std::cout << "Generate rating matrices"<<std::endl;
    int64 startTime1 = LogTimeMM::getSystemTime();
    static SpMat trainMatrix(userCount, itemCount);
    static SpMat_R trainMatrix_R(userCount, itemCount);
    
    static std::vector<Rating> testRatings;
   
//    tripletList.reserve(estimation_of_entries);
     std::vector<T> tripletList;
    for ( int u = 0; u <userCount; u++){
        std::vector<Rating> rating = user_ratings[u];
        for (int i = rating.size() - 1; i >= 0; i--) {
            user_id = rating[i].userId;
            item_id = rating[i].itemId;
            if (i == rating.size() - 1) { // test
                testRatings.push_back(rating[i]);
            } else { // train
                tripletList.push_back(T(user_id,item_id, 1));
            }
//                trainMatrix.insert(user_id, item_id) =  1;
        }
    }
    trainMatrix.setFromTriplets(tripletList.begin(), tripletList.end());
    trainMatrix_R.setFromTriplets(tripletList.begin(), tripletList.end());
//    trainMatrix.makeCompressed();
    std::cout << "Generated splitted matrices time:"<< LogTimeMM::getSystemTime() - startTime1 << std::endl;
    std::cout << "data\t"<< dataset_name<<std::endl;
    std::cout << "#Users\t"<< userCount<<std::endl;
    std::cout << "#items\t"<< itemCount<<std::endl;
    std::cout << "#Ratings\t"<< trainMatrix.cols()<<"\t"<< "tests\t"<<testRatings.size()<< std::endl;
    
    std::cout <<"=========================================="<<std::endl;
    
//    ItemPopularity popularity = new ItemPopularity(trainMatrix, testRatings, topK, threadNum);
//    evaluate_model(popularity, "Popularity");

//    long start = System.currentTimeMillis();
//    model.buildModel();
//    public void buildModel() {
//        for (int i = 0; i < itemCount; i++) {
//            // Measure popularity by number of reviews received.
//            item_popularity[i] = trainMatrix.getColRef(i).itemCount();
//        }
//    }
//    double item_popularity[itemCount];
//    for (int i = 0; i < itemCount; i++) {
//        item_popularity[i] = trainMatrix.outerIndexPtr()[i+1] - trainMatrix.outerIndexPtr()[i];
//
//    }
//    model.evaluate(utestRatings);
    assert (userCount == testRatings.size());
    for (int u = 0; u < userCount; u++)
        assert( u == testRatings[u].userId);

//    下面的用多线程splited by users, here用cuda

//    MF_fastALS(SpMat trainMatrix, std::vector<Rating> testRatings,
//               int topK, int factors, int maxIter, double w0, double alpha, double reg, double init_mean, double init_stdev, bool showprogress, bool showloss);
    
    MF_fastALS als(trainMatrix, trainMatrix_R, testRatings, topK, factors, maxIter, w0, alpha, reg, init_mean,init_stdev, showprogress,showloss);
    std::cout<<"success"<<std::endl;
//    evaluate_model(als, "MF_ALS")
//    public static double[] evaluate_model(TopKRecommender model, String name) {
//        long start = System.currentTimeMillis();
//        model.buildModel();
//        model.evaluate(testRatings);
//
//        double[] res = new double[3];
//        res[0] = model.hits.mean();
//        res[1] = model.ndcgs.mean();
//        res[2] = model.precs.mean();
//        System.out.printf("%s\t <hr, ndcg, prec>:\t %.4f\t %.4f\t %.4f [%s]\n",
//                          name, res[0], res[1], res[2],
//                          Printer.printTime(System.currentTimeMillis() - start));
//        return res;
//    }
    
    
    als.buildModel();
    for ( int u = 0 ; u < userCount; u++){
        double * result = new double[3];
        int gtItem = testRatings[u].itemId;
        result = als.evaluate_for_user(u, gtItem,topK);
        hits[u] = result[0];
        ndcgs[u] = result[1];
        precs[u] = result[2];
//        delete[] result;
    }
    double res [3];
    res[0] = hits.mean();
    res[1] = ndcgs.mean();
    res[2] = precs.mean();
    
    std::cout << "hr, ndcg, prec:"<<res[0]<< "\t" << res[1]<<"\t" <<res[2]<<std::endl;
}

    



