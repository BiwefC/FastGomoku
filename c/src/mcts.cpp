#include "mcts.hpp"

using namespace gomoku;


mcts::MCTS::MCTS(ChessState *root, PyEvaluator *evaluator, bool dirichlet) :
    evaluator(evaluator),
    root(nullptr),
    rnd_eng(std::time(nullptr)),
    rnd_dis(0, 1),
    dirichlet_noise(dirichlet)
{
    set_root(root);
    states.reserve(C_EVAL_BATCHSIZE);
}

mcts::MCTS::MCTS(ChessState *root, PyEvaluator *evaluator, bool dirichlet, int seed) :
    evaluator(evaluator),
    root(nullptr),
    rnd_eng(seed),
    rnd_dis(0, 1),
    dirichlet_noise(dirichlet)
{
    set_root(root);
    states.reserve(C_EVAL_BATCHSIZE);
}

Position mcts::MCTS::get_simulation_result(double temp)
{
    SearchedProb prob = {0.0f};
    root->get_searched_prob(prob, temp);
    double r = rnd_dis(rnd_eng);
    int index = 0;
    float accum = 0.0f;

    for (int i = 0; i < BOARD_SIZE*BOARD_SIZE; i++) {
        accum += prob[i];
        if (accum > r) {
            index = i;
            break;
        }
    }
    Position pos = index2pos(index);
    do_set(pos);
    return pos;
}

void mcts::MCTS::do_set(Position pos)
{
    bool found = false;
    if (root->expanded) {
        int index = 0;
        for (auto &a : root->child_actions) {
            if (a.pos == pos) {
                found = true;
                break;
            }
            index++;
        }
        if (found) {
            do_set(index);
        }
    }
    if (!found) {
        if (root->game.get(pos) == COLOR_NONE) {
            ChessState *state = new ChessState(nullptr, root->game, -root->color);
            state->game.move(root->color, pos);
            step_finish(state);

        }
    }
    return;
}

void mcts::MCTS::do_set(int action_index)
{
    auto &a = root->child_actions[action_index];
    if (!a.expanded) a.expand();
    ChessState *state = a.child_state;
    a.stepped = true;
    step_finish(state);
}

void mcts::MCTS::step_finish(ChessState *state)
{
    set_root(state);
    steps++;
}

void mcts::MCTS::set_root(ChessState *state)
{
    delete root;
    state->parent_action = nullptr;

    root = state;
}

void mcts::MCTS::search(int k)
{
    #ifdef DEBUG
    clock_t start = clock();
    #endif
    for (int t = 0; t < k / C_EVAL_BATCHSIZE; t++) {
        states.clear();
        batch_finish = false;
        for (int i_batch = 0; i_batch < C_EVAL_BATCHSIZE; i_batch++) {
            simulate_once();
        }
        std::vector<Game*> games;
        std::vector<Color> pov;
        games.reserve(states.size());
        for (auto state : states) {
            games.emplace_back(&state->game);
            pov.emplace_back(state->color);
        }
        auto evals = evaluator->evaluate(games, pov);
        for (int i = 0; i < states.size(); i++) {
            states[i]->set_eval(evals[i]);
        }
    }
    #ifdef DEBUG
    clock_t ends = clock();
    std::cout <<"Simulate Time : "<<(double)(ends - start)/ CLOCKS_PER_SEC << std::endl;
    #endif
}

int mcts::MCTS::simulate_once()
{
    auto *current = root;

    while (true) {

        if (batch_finish) {
            return 0;
        }

        if (current->evaluating) {
            batch_finish = true;
            return 0;
        }

        current->visit_count++;
        if (current->game.is_over) {
            current->backprop_value(); // game over, backprob {+1, 0, -1} directly
            return 0;
        }
        if (!current->expanded) {
            if (current->evaluated) {
                current->expand();
            }
            else {
                states.emplace_back(current);
                current->evaluating = true;
                return 0;
            }
        }

        if (root == current && dirichlet_noise && root->expanded && !(root->noise_applied)) {
            current->apply_dirichlet_noise(0.04f, 0.25f, seed);
        }

        if (current->expanded) {
            if (current->child_actions.size() == 0) {
                return 0;
            }
            auto &actions = current->child_actions;
            int max_i = -1;
            float max_ucb = 0, ucb = 0;
            for (int i = 0; i < actions.size(); i++) {
                ucb = actions[i].get_ucb();
                if (ucb > max_ucb || max_i == -1) {
                    max_i = i;
                    max_ucb = ucb;
                }
            }
            auto &action = actions[max_i];
            if (!action.expanded) {
                action.expand();
            }
            current = action.child_state;
        }
    }
}
