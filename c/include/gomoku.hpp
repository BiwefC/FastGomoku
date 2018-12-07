#pragma once

#include <iostream>
#include <sstream>
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
        return get(pos.x, pos.y);
    }
    inline Color get(int x, int y) {
        if (x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE) {
            return board[x][y];
        }
        else {
            return 0;
        }
    }
    Game();
    void move(Color color, Position pos);
    void move(Color color, int x, int y);
    void graphic();
    void show_result();
    bool is_swappable(); // define the swap rule here. Swap1 & Swap2 are surpported.
//private:
    void check_is_over(int x, int y);
    bool is_legal_move(Color color, Position pos);
    void get_observation(Observation &obsv, Color pov);

private:
    Position last_move;
    Judge judge;
    int steps;
};

}