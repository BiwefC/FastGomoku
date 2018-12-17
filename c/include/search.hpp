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
#include "eval.hpp"

using namespace eval;


namespace search
{

const float C_PUCT = 3.0;
const int C_EVAL_BATCHSIZE = 32;
const float C_PROB_THRESHOLD = 1E-5F;

typedef int8_t Player;
const Player PLAYER_1 = 1;
const Player PLAYER_2 = -1;

typedef float SearchedProb[BOARD_SIZE * BOARD_SIZE];


class State;

class Action {
public:
    bool expanded;
    bool stepped;
    float prior_prob;
    float action_value;
    State *parent_state;
    State *child_state;
    Position pos;
    Action(State *parent, Position pos, float prob);
    ~Action();
    float get_ucb();
    int get_visit_count();
    void expand();
private:
    float _ucb;
};

class State {
public:
    Game game;
    int visit_count;
    Color color; // color going to play

    float sum_value;
    float value;
    bool expanded;
    bool evaluated;
    bool evaluating;
    bool noise_applied;
    Action *parent_action;

    bool swappable;
    std::vector<Action> child_actions;

    State(Action *parent, Game game, Color color);
    ~State();
    void set_eval(Evaluation *e);
    void expand();
    void apply_dirichlet_noise(float alpha, float epsilon, int seed);
    void backprop_value();
    void refresh_value();
    void get_searched_prob(SearchedProb &prob, double temp);
private:
    Evaluation *eval;

};

class MCTS {
public:
    State *root;
    Evaluator *evaluator;
    int steps;
    MCTS(State *root, Evaluator *evaluator, bool dirichlet);
    MCTS(State *root, Evaluator *evaluator, bool dirichlet_noise, int seed);
    Position get_step(double temp);
    void step(Position pos);
    void simulate(int k);

private:
    bool dirichlet_noise;
    void step(int action_index);
    void step_finish(State* state);
    void set_root(State* state);
    std::default_random_engine rnd_eng;
    std::uniform_real_distribution<double> rnd_dis;
    int simulate_once();
    std::vector<State*> states;
    bool batch_finish;
    int seed;
};

}