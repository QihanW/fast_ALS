//
//  time.cpp
//  fast_ALS
//
//  Created by yangsong on 5/4/19.
//  Copyright © 2019 yangsong. All rights reserved.
//

//#include "time.hpp"
#ifndef TIME_H_
#define TIME_H_

#include <stdio.h>
#include <sys/time.h>
typedef long long int64;

class LogTimeMM
{
public:
    static int64 getSystemTime(){
        struct timeval tv;                //获取一个时间结构
        gettimeofday(&tv, NULL);   //获取当前时间
        int64 t = tv.tv_sec;
        t *=1000;
        t +=tv.tv_usec/1000;
        return  t;
    }
};

#endif TIME_H_
