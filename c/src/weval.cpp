#include "weval.hpp"

using namespace eval;
using namespace search;
using namespace std;
using namespace chrono;


void weval::run(char *w0, char *w1, int seed, char *outdir, int v_level)
{
    PyEvaluator eval[2] = {PyEvaluator(w0),
                           PyEvaluator(w1)};

    Game game;

    int p_first = seed % 2;
    Color p0_color;
    if (p_first == 0) {
        p0_color = COLOR_BLACK;
    }
    else {
        p0_color = COLOR_WHITE;
    }

    MCTS mcts[2] = {MCTS(new State(nullptr, Game(), COLOR_BLACK), &eval[0], false, seed),
                    MCTS(new State(nullptr, Game(), COLOR_BLACK), &eval[1], false, seed)};

    int p = p_first;
    Color c = COLOR_BLACK;
    while (!game.is_over) {

        mcts[p].simulate(1600);
        Position pos = mcts[p].random_step(0.0);
        mcts[1 - p].step(pos);


        game.move(c, pos);
        if (v_level) {
            game.graphic();
        }
        game.check_is_over(pos.x, pos.y);
        c = -c;
        p = 1 - p;
    }
    game.show_result();
    string result;
    if (game.winner == p0_color) {
        result = "W";
    }
    else if (game.winner == COLOR_NONE) {
        result = "D";
    }
    else {
        result = "L";
    }

    PyObject_CallMethod(py_util::module, "save_eval_result", "ss", outdir, result.c_str());
}
