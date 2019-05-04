//
//  rating.cpp
//  fast_ALS
//
//  Created by yangsong on 5/1/19.
//  Copyright © 2019 yangsong. All rights reserved.
//

#include <stdio.h>
#include <iostream>
using namespace std;
class rating
{
public:
    int userId;
    int itemId; // item id, starts from 0
    float score;
    long timestamp;
    rating(int userId, int itemId, float score, long timest);
};

rating::rating(int userId, int itemId, float score, long timestamp){
    userId = userId;
    itemId = itemId;
    score = score;
    timestamp = timestamp;
    
}
