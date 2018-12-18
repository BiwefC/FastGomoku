#include "weval.hpp"

using namespace eval;
using namespace search;
using namespace std;
using namespace chrono;


void weval::run(char *w0, char *w1, int seed, char *outdir, int v_level, int step)
{
    PyEvaluator eval[2] = {PyEvaluator(w0),
                           PyEvaluator(w1)};

    Game game;
    py_api::init_py_util();

    int p_first = seed % 2;
    Color p0_color;
    if (p_first == 0) {
        p0_color = COLOR_BLACK;
    }
    else {
        p0_color = COLOR_WHITE;
    }

    MCTS mcts[2] = {MCTS(new ChessState(nullptr, Game(), COLOR_BLACK), &eval[0], false, seed),
                    MCTS(new ChessState(nullptr, Game(), COLOR_BLACK), &eval[1], false, seed)};

    int p = p_first;
    Color c = COLOR_BLACK;
    while (!game.is_over) {

        mcts[p].simulate(step);
        Position pos = mcts[p].get_step(0.0);
        mcts[1 - p].move_step(pos);


        game.move(c, pos);
        if (v_level) {
            game.graphic();
        }
        game.check_is_over(pos.row, pos.col);
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

    PyObject_CallMethod(py_api::module, "save_eval_result", "ss", outdir, result.c_str());
}
