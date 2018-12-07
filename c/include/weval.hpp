#pragma once

#include <fstream>
#include <chrono>
#include "eval.hpp"
#include "search.hpp"
#include "gomoku.hpp"
#include "py_util.hpp"


namespace weval
{

void run(char *w0, char *w1, int rounds, char *outdir);

}
