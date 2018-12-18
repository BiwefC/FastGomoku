#pragma once

#include <iostream>
#include <sstream>
#include <cstring>
#include "base.hpp"
#include "judge.hpp"


namespace gomoku
{

class Game {
public:
    Board board;
    bool is_over;
    Color winner;
    inline Color get(Position pos) {
        return get(pos.row, pos.col);
    }
    inline Color get(int row, int col) {
        if (row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE) {
            return board[row][col];
        }
        else {
            return 0;
        }
    }
    Game();
    void move(Color color, Position pos);
    void move(Color color, int row, int col);
    void graphic();
    void show_result();
    void check_is_over(int row, int col);
    bool is_legal_move(Color color, Position pos);
    void get_observation(Observation &obsv, Color pov);

private:
    Position last_move;
    Judge judge;
    int steps;
};

}
