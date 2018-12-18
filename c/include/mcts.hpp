#pragma once

#include <random>
#include <cmath>
#include <list>
#include <random>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <iterator>
#include "gomoku.hpp"
#include "search.hpp"
#include "eval.hpp"

using namespace eval;
using namespace search;

namespace mcts
{

class MCTS {
public:
    ChessState *root;
    PyEvaluator *evaluator;
    int steps;
    MCTS(ChessState *root, PyEvaluator *evaluator, bool dirichlet);
    MCTS(ChessState *root, PyEvaluator *evaluator, bool dirichlet_noise, int seed);
    Position get_simulation_result(double temp);
    void do_set(Position pos);
    void search(int k);

private:
    bool dirichlet_noise;
    void do_set(int action_index);
    void step_finish(ChessState* state);
    void set_root(ChessState* state);
    std::default_random_engine rnd_eng;
    std::uniform_real_distribution<double> rnd_dis;
    int simulate_once();
    std::vector<ChessState*> states;
    bool batch_finish;
    int seed;
};

}
