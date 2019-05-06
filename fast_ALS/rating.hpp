//
//  rating.cpp
//  fast_ALS
//
//  Created by yangsong on 5/1/19.
//  Copyright © 2019 yangsong. All rights reserved.
//

#ifndef RATING_H_
#define RATING_H_

#include <stdio.h>
#include <iostream>
#include <string>

//class Rating {
//public:
//    int userId;
//    int itemId; // item id, starts from 0
//    float score;
//    long timestamp;
//
//    Rating() = default;
//    Rating(std::string x, std::string y, std::string z, std::string q){
//        userId = stoi(x);
//        itemId = stoi(y);
//        score  = stof(z);
//        timestamp = stol(q);}
//};

struct Rating {
    int userId;
    int itemId; // item id, starts from 0
    float score;
    long timestamp;
    
    Rating() = default;
//    Rating(std::string x, std::string y, std::string z, std::string q){
//        userId = stoi(x);
//        itemId = stoi(y);
//        score  = stof(z);
//        timestamp = stol(q);}
    Rating(int _userId,
           int _itemId,
           float _score,
           long _timestamp) :
            userId(_userId),
            itemId(_itemId),
            score(_score),
            timestamp(_timestamp) { }
};

#endif


