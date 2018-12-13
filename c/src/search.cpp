#include "search.hpp"

using namespace gomoku;

search::MCTS::MCTS(State *root, Evaluator *evaluator, bool dirichlet, int seed) :
    evaluator(evaluator),
    root(nullptr),
    rnd_eng(seed),
    rnd_dis(0, 1),
    //current_color(COLOR_BLACK),
    dirichlet_noise(dirichlet)
{
    set_root(root);
}

Position search::MCTS::random_step(double temp)
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
    step(pos);
    return pos;
}

void search::MCTS::step(Position pos)
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
            step(index);
        }
    }
    if (!found) {
        if (root->game.get(pos) == COLOR_NONE) {
            State *state = new State(nullptr, root->game, -root->color);
            state->game.move(root->color, pos);
            step_finish(state);

        }
    }
    return;
}

void search::MCTS::step(int action_index)
{
    auto &a = root->child_actions[action_index];
    if (!a.expanded) a.expand();
    State *state = a.child_state;
    a.stepped = true;
    step_finish(state);
}

void search::MCTS::step_finish(State *state)
{
    set_root(state);
    steps++;
}

void search::MCTS::set_root(State *state)
{
    delete root;
    state->parent_action = nullptr;

    root = state;
}

void search::MCTS::simulate(int k)
{
    #ifdef DEBUG
    clock_t start = clock();
    #endif
    for (int t = 0; t < k / C_EVAL_BATCHSIZE; t++) {
        states.clear();
        // std::vector<State*> states;
        states.reserve(C_EVAL_BATCHSIZE);
        batch_finish = false;
        for (int i_batch = 0; i_batch < C_EVAL_BATCHSIZE; i_batch++) {
            // int a = simulate_once();
            #ifdef MULTI_THREAD
            thread t(&search::MCTS::simulate_once, this);
            t.join();
            #else
            simulate_once();
            #endif

            // if (a != 0) {
                // goto batch_ready;
            // }
        }
    // batch_ready:
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
        // call evaluator
        // assign evaluations
    }
    #ifdef DEBUG
    clock_t ends = clock();
    std::cout <<"Simulate Time : "<<(double)(ends - start)/ CLOCKS_PER_SEC << std::endl;
    #endif
}

int search::MCTS::simulate_once()
{
    auto *current = root;

    while (true) {
        #ifdef MULTI_THREAD
        current->lock.lock();
        #endif

        if (batch_finish) {
            #ifdef MULTI_THREAD
            current->lock.unlock();
            #endif
            return 0;
        }

        if (current->evaluating) {
            batch_finish = true;
            #ifdef MULTI_THREAD
            current->lock.unlock();
            #endif
            return 0;
        }
        // return 1;

        current->visit_count++;
        if (current->game.is_over) {
            current->backprop_value(); // game over, backprob {+1, 0, -1} directly
            #ifdef MULTI_THREAD
            current->lock.unlock();
            #endif
            return 0;
        }
        if (!current->expanded) {
            if (current->evaluated) {
                current->expand();
            }
            else {
                states.emplace_back(current);
                current->evaluating = true;
                #ifdef MULTI_THREAD
                current->lock.unlock();
                #endif
                return 0;
            }
        }

        if (root == current && dirichlet_noise && root->expanded && !(root->noise_applied)) {
            current->apply_dirichlet_noise(0.04f, 0.25f, seed);
        }

        if (current->expanded) {
            if (current->child_actions.size() == 0) {
                #ifdef MULTI_THREAD
                current->lock.unlock();
                #endif
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
            #ifdef MULTI_THREAD
            current->lock.unlock();
            #endif
            current = action.child_state;
        }
    }
}

search::State::State(Action *parent, Game game, Color color)
    : parent_action(parent),
    color(color),
    visit_count(0),
    sum_value(0.0f),
    eval(nullptr),
    evaluating(false),
    evaluated(false),
    expanded(false),
    game(game),
    child_actions(),
    value(0.0f),
    noise_applied(false)
{
    swappable = game.is_swappable();
}

search::State::~State()
{
    delete eval;
}

void search::State::set_eval(Evaluation *e)
{
    eval = e;
    refresh_value();
    backprop_value();
    evaluating = false;
    evaluated = true;
}

void search::State::backprop_value()
{
    auto current = this;
    Action *action;
    float dv;
    while (true) {

        if (swappable) {
            dv = -std::fabs(value); // any imbalance is in favor of the swapper
        }
        else {
            if (color == current->color) {
                dv = value;
            }
            else {
                dv = -value;
            }
        }
        current->sum_value += dv;

        action = current->parent_action;
        if (action == nullptr) break;

        if (action->parent_state->color == current->color) {
            action->action_value = current->sum_value / current->visit_count;
        }
        else {
            action->action_value = -current->sum_value / current->visit_count;
        }
        current = action->parent_state;
    }
}

void search::State::refresh_value()
{
    if (game.is_over) {
        if (game.winner == color) {
            value = 1;
        }
        else if (game.winner == -color) {
            value = -1;
        }
        else {
            value = 0;
        }
    }
    else if (eval != nullptr) {
        value = eval->value;
    }
}

void search::State::apply_dirichlet_noise(float alpha, float epsilon, int seed)
{
    // if (!expanded) expand;
    static std::gamma_distribution<float> gamma(alpha, 1.0f);
    static std::default_random_engine rng(seed);

    float d_sum = 0;
    auto dirichlet_vector = std::vector<float>();
    auto unconsidered_pos = std::vector<Position>();

    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            auto pos = Position{i, j};
            if (game.is_legal_move(color, pos)) {
                float d = gamma(rng);
                d_sum += d;
                dirichlet_vector.emplace_back(d);

                bool found = false;
                for (auto &child : child_actions) {
                    if (child.pos == pos) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    unconsidered_pos.emplace_back(pos);
                }
            }
        }
    }

    for (int i = 0; i < child_actions.size(); i++) {
        float dir = dirichlet_vector[i + unconsidered_pos.size()] / d_sum;
        auto &child = child_actions[i];
        child.prior_prob = (1 - epsilon) * child.prior_prob + epsilon * dir;
    }

    for (int i = 0; i < unconsidered_pos.size(); i++) {
        float p = dirichlet_vector[i] / d_sum * epsilon;
        if (p >= C_PROB_THRESHOLD) {
            child_actions.emplace_back(this, unconsidered_pos[i], p);
        }
    }

    noise_applied = true;
}

void search::State::expand()
{
    if (game.is_over) expanded = true;
    if (expanded) return;

    float sum = 0;
    int count = 0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (game.get(i, j) == COLOR_NONE) {
                sum += eval->policy[pos2index(i, j)];
                count++;
            }
        }
    }

    child_actions.reserve(count);

    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            float p = eval->policy[pos2index(i, j)] / sum;
            if (game.get(i, j) == COLOR_NONE && p >= C_PROB_THRESHOLD) {
                child_actions.emplace_back(this, Position{i, j}, p);
            }
        }
    }

    delete eval;
    eval = nullptr;
    expanded = true;

}

void search::State::get_searched_prob(SearchedProb &prob, double temp)
{
    bool use_max = false;
    double power = 0;
    int index = 0;
    if (temp == 0.0) {
        use_max = true;
    }
    else {
        power = 1 / temp;
        if (power >= 100) use_max = true;
    }
    if (use_max) {
        int max = 0;
        int max_count = 0;
        for (int i = 0; i < child_actions.size(); i++) {
            int v = child_actions[i].get_visit_count();
            if (v > max) {
                max = v;
                max_count = 1;
            }
            else if (v == max) {
                max_count++;
            }
        }
        for (int i = 0; i < BOARD_SIZE*BOARD_SIZE; i++) {
            prob[i] = 0;
        }
        for (int i = 0; i < child_actions.size(); i++) {
            int v = child_actions[i].get_visit_count();
            if (v == max) {
                prob[pos2index(child_actions[i].pos)] = 1.0 / max_count;
            }
        }

    }
    else {
        double sum = 0;
        double prob_power[BOARD_SIZE * BOARD_SIZE] = {0.0};
        for (auto &action : child_actions) {
            double p = std::pow(action.get_visit_count(), power);
            prob_power[pos2index(action.pos)] = p;
            sum += p;
        }
        for (int i = 0; i < BOARD_SIZE*BOARD_SIZE; i++) {
            prob[i] = prob_power[i] / sum;
        }
    }
}


search::Action::Action(State *parent_state, Position pos, float prob) :
    expanded(false),
    stepped(false),
    prior_prob(prob),
    parent_state(parent_state),
    child_state(nullptr),
    pos(pos),
    action_value(0.0f)
{
    ;
}


search::Action::~Action()
{
    if (!stepped) {
        delete child_state;
    }
}

float search::Action::get_ucb()
{
    //static auto rnd_eng = std::default_random_engine(std::time(nullptr));
    //static auto rnd_dis = std::uniform_real_distribution<float>(0, 1E-3F);
    float u = 0;
    if (child_state != nullptr && child_state->evaluating) {
        u = -100; // apply a virtual loss
    }
    u += action_value +
        C_PUCT * prior_prob * std::sqrt(parent_state->visit_count) / (1 + get_visit_count()); //+ rnd_dis(rnd_eng);
    _ucb = u; // cache ucb for debugging
    return u;
}

void search::Action::expand()
{
    if (expanded) return;
    State *state = new State(this, parent_state->game, -parent_state->color);
    state->game.move(parent_state->color, pos);
    state->refresh_value();
    child_state = state;
    expanded = true;
}

int search::Action::get_visit_count()
{
    return (child_state == nullptr ? 0 : child_state->visit_count);
}
