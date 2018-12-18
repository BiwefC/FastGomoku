#pragma once

#include <cinttypes>


namespace gomoku
{

const bool C_SIX_WIN = true;
const int BOARD_SIZE = 15;

typedef int8_t Color;

const Color COLOR_WHITE = -1;
const Color COLOR_NONE = 0;
const Color COLOR_BLACK = 1;

typedef Color Board[BOARD_SIZE][BOARD_SIZE];
typedef float Observation[2][BOARD_SIZE][BOARD_SIZE];

struct Position {
    int row;
    int col;
    bool operator==(const Position &p1) { return row == p1.row && col == p1.col; };
};

inline Position index2pos(int index) {
    return Position{index / BOARD_SIZE, index % BOARD_SIZE};
}
inline int pos2index(Position pos) {
    return pos.row * BOARD_SIZE + pos.col;
}
inline int pos2index(int row, int col) {
    return row * BOARD_SIZE + col;
}

}