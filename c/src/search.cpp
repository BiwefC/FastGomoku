#include "search.hpp"

using namespace gomoku;


search::ChessState::ChessState(Move *parent, Game game, Color color)
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
    ;
}

search::ChessState::~ChessState()
{
    delete eval;
}

void search::ChessState::set_eval(Evaluation *e)
{
    eval = e;
    refresh_value();
    backprop_value();
    evaluating = false;
    evaluated = true;
}

void search::ChessState::backprop_value()
{
    auto current = this;
    Move *action;
    float dv;
    while (true) {
        if (color == current->color) {
            dv = value;
        }
        else {
            dv = -value;
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

void search::ChessState::refresh_value()
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

void search::ChessState::apply_dirichlet_noise(float alpha, float epsilon, int seed)
{
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

void search::ChessState::expand()
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

void search::ChessState::get_searched_prob(SearchedProb &prob, double temp)
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

	memset(prob, 0, sizeof(float) * BOARD_SIZE * BOARD_SIZE);

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


search::Move::Move(ChessState *parent_state, Position pos, float prob) :
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


search::Move::~Move()
{
    if (!stepped) {
        delete child_state;
    }
}

float search::Move::get_ucb()
{
    float u = 0;
    if (child_state != nullptr && child_state->evaluating) {
        u = -100; // apply a virtual loss
    }
    u += action_value +
        C_PUCT * prior_prob * std::sqrt(parent_state->visit_count) / (1 + get_visit_count()); //+ rnd_dis(rnd_eng);
    _ucb = u; // cache ucb for debugging
    return u;
}

void search::Move::expand()
{
    if (expanded) return;
    ChessState *state = new ChessState(this, parent_state->game, -parent_state->color);
    state->game.move(parent_state->color, pos);
    state->refresh_value();
    child_state = state;
    expanded = true;
}

int search::Move::get_visit_count()
{
    return (child_state == nullptr ? 0 : child_state->visit_count);
}
