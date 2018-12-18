#include "selfplay.hpp"


double get_temp(int i_step) {
    if (i_step <= 8) {
        return -0.125 * i_step + 1.0;
    }
    else {
        return 0.0;
    }
}
int selfplay_init_numpy() {
    if (PyArray_API == NULL) {
        import_array();
    }
}
void selfplay::save_samples(vector<StepSample> &samples) {
    selfplay_init_numpy();

    int count = samples.size();

    auto obsvs = new Observation[count];
    auto probs = new SearchedProb[count];
    auto results = new int[count];

    int i = 0;
    for (auto &sample : samples) {
        memcpy(&obsvs[i], &sample.observation, sizeof(Observation));
        memcpy(&probs[i], &sample.prob, sizeof(SearchedProb));
        results[i] = sample.result;
        i++;
    }

    npy_intp dims_obsvs[4] = {count, 2, BOARD_SIZE, BOARD_SIZE};
    npy_intp dims_probs[2] = {count, BOARD_SIZE * BOARD_SIZE};
    npy_intp dims_results[1] = {count};

    auto py_obsvs = PyArray_SimpleNewFromData(4, dims_obsvs, NPY_FLOAT, obsvs);
    auto py_probs = PyArray_SimpleNewFromData(2, dims_probs, NPY_FLOAT, probs);
    auto py_results = PyArray_SimpleNewFromData(1, dims_results, NPY_INT, results);

    PyObject_CallMethod(py_api::module, "save_samples", "OOO", py_obsvs, py_probs, py_results);

    Py_XDECREF(py_obsvs);
    Py_XDECREF(py_probs);
    Py_XDECREF(py_results);

    delete[] obsvs;
    delete[] probs;
    delete[] results;
}

void selfplay::run(char* weight, int v_level, int seed) {
    auto evaluator = new PyEvaluator(weight);
    vector<StepSample> samples;
    int result_cursor = 0;
    Game game;
    py_api::init_py_util();

    auto state = new ChessState(nullptr, game, COLOR_BLACK);
    if (-1 == seed) {
        seed = std::time(nullptr);
    }
    auto mcts = mcts::MCTS(state, evaluator, true, seed);
    int i_step = 0;
    while (true) {
        i_step++;
        double temp = get_temp(i_step);

        if (mcts.root->game.is_over) {
            mcts.root->game.show_result();

            for (; result_cursor < samples.size(); result_cursor++) {
                auto &sample = samples[result_cursor];
                if (mcts.root->game.winner == sample.color) {
                    sample.result = 1;
                }
                else if (mcts.root->game.winner == COLOR_NONE) {
                    sample.result = 0;
                }
                else {
                    sample.result = -1;
                }
            }

            break;
        }
        StepSample sample;
        sample.color = mcts.root->color;

        mcts.search(1600);

        mcts.root->game.get_observation(sample.observation, mcts.root->color);
        mcts.root->get_searched_prob(sample.prob, temp);
        samples.emplace_back(sample);

        mcts.get_simulation_result(temp);
        if (v_level) {
            mcts.root->game.graphic();
        }

    }
    save_samples(samples);
    return;
}
