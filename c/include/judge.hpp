#pragma once

#include<iostream>
#include<iomanip>
#include<vector>
#include "base.hpp"
using namespace std;

namespace gomoku
{

struct p_tmp
{
    int i;
    int j;
};

class Judge
{
public:
    Judge();

    Color do_judge(Board &table, int x, int y);
/*  int chongsan( Table T );
    int huoer( Table T );
    int chonger( Table T );
    int huoyi( Table T );
    int chongyi( Table T );*/
private:
    int color ;
    int anticolor ;

    vector<p_tmp> sanArr1;  // -
    vector<p_tmp> sanArr2;  // |
    vector<p_tmp> sanArr3;  // \

    vector<p_tmp> sanArr4;  // /

    vector<p_tmp> siArr1;
    vector<p_tmp> siArr2;
    vector<p_tmp> siArr3;
    vector<p_tmp> siArr4;

    bool longlink(Board &table);
    int fivelink(Board &table, Color color);
    int si(Board &table, char line, char column);
    int huosan(Board &table, char line, char column);
};

}