#pragma once

#include <iostream>
#include <numeric>
#include <ctime>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "gomoku.hpp"
#include "mcts.hpp"
#include "py_api.hpp"

using namespace std;
using namespace search;
using namespace eval;
using namespace gomoku;

namespace selfplay
{

struct StepSample {
    Observation observation;
    SearchedProb prob;
    Color color;
    int result;
};

void save_samples(vector<StepSample> &samples);
void run(char* weight, int v_level, int seed);

};
