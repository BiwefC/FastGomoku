#pragma once

#include <vector>
#include <numpy/arrayobject.h>
#include <Python.h>
#include "gomoku.hpp"

using namespace gomoku;
namespace eval
{

typedef float Policy[BOARD_SIZE * BOARD_SIZE];

struct Evaluation {
    Policy policy;
    float value;
};

class PyEvaluator {
public:
    bool init_succeeded;
    PyEvaluator(char *weight);
    ~PyEvaluator();
    std::vector<Evaluation*> evaluate(std::vector<Game*> games, std::vector<Color> pov);
private:
    PyObject *py_network;
};

}
