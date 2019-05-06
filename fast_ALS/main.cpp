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
#include "fast_ALS.hpp"
//#include "rating.hpp"


int main(int argc, const char * argv[])
{
    std::string dataset_name = "yelp";
    std::string method = "FastALS";
    double w0 = 10;
    bool showProgress = false;
    bool showLoss = true;
    int factors = 64;
    int maxIter = 500;
    double reg = 0.01;
    double alpha = 0.75;
    
    //处理原始数据
    //ReadRatings_GlobalSplit()
    //为什么是static?
    static int userCount;
    static int itemCount;
    std::cout <<"Holdone out splitting" <<std::endl;
    std::cout << "Sort items for each user."<<std::endl;
    int64 startTime = LogTimeMM::getSystemTime();
    std::vector<std::vector<Rating>> user_ratings;
    char input_filename[] = "/Users/yangsong/Desktop/fast_ALS/data/yelp.rating";
    std::ifstream  fin(input_filename);
    if (!fin.is_open()) {
        fprintf(stderr, "Error: cannot open the file %s\n", input_filename);
        exit(EXIT_FAILURE);
    }
//    char line[1024]={0};
//    std::string  x = "";
//    std::string  y = "";
//    std::string  z = "";
//    std::string  q = "";
    std::string line;
    int user_id;
    int item_id;
    float score;
    long timestamp;

    std::cout<<"dsds"<<std::endl;
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
    }
    
    { // test
//        std::ofstream fout("/Users/yangsong/Desktop/fast_ALS/data/output.txt");
        FILE *fout = fopen("/Users/yangsong/Desktop/fast_ALS/data/output.txt", "w");
        for (const auto &user_vector : user_ratings) {
            for (const auto & rating : user_vector) {
//                fout << rating.userId << "\t"
//                << rating.itemId << "\t"
//                << rating.score << "\t"
//                << rating.timestamp << std::endl;
                fprintf(fout,"%d\t%d\t%.1f\t%ld\n",
                                    rating.userId,
                                    rating.itemId,
                                    rating.score,
                                    rating.timestamp);
            }
        }
        fclose(fout);
    }

    
//    while(fin.getline(line, sizeof(line)))
//    {
//        std::stringstream  word(line);
//        word >> x;
//        word >> y;
//        word >> z;
//        word >> q;
//        Rating rating(std::string x, std::string y, std::string z, std::string q);
//        std::cout << rating.userId;
//        if (user_ratings.size() < rating.userId + 1){
////            vector<Rating> temp;
//
//            user_ratings.push_back(std::vector<Rating>());
//            user_ratings.rbegin() -> push_back(rating);
//        }
//    }
}
