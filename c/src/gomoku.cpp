#include "gomoku.hpp"

using namespace gomoku;
using namespace std;


Game::Game() : is_over(false), winner(COLOR_NONE), last_move{-1, -1}, steps(0)
{
    memset(board, COLOR_NONE, sizeof(Color) * BOARD_SIZE * BOARD_SIZE);
}

void Game::move(Color p, Position pos)
{
    move(p, pos.row, pos.col);
}

void Game::move(Color p, int row, int col)
{
    if (!is_legal_move(p, Position{row,col})) return;
    board[row][col] = p;
    check_is_over(row, col);
    last_move = Position{row, col};
    steps++;
}

void Game::graphic()
{
    ostringstream out;
    out << "step" << steps << endl;
    const int cell_size = 3;
    const int row_size = (cell_size + 1) * BOARD_SIZE + 1;
    char line[row_size + 1];
    char line2[row_size + 1];
    for (int i = 0; i < row_size; i++) {
        if (i % (cell_size + 1) == 0)
            line[i] = ' ';
        else
            line[i] = ' ';

        line2[i] = ' ';
    }
    line[row_size] = '\0';
    line2[row_size] = '\0';
    out << line << endl;
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            line2[j * (cell_size + 1)] = ' ';
            if (last_move.row == i) {
                if (last_move.col == j) {
                    line2[j * (cell_size + 1)] = '[';
                }
                else if (last_move.col == j - 1) {
                    line2[j * (cell_size + 1)] = ']';
                }
            }

            int st = j * (cell_size + 1) + cell_size / 2 + 1;
            if (board[i][j] == COLOR_BLACK)
                line2[st] = 'X';
            else if (board[i][j] == COLOR_WHITE)
                line2[st] = 'O';
            else
                line2[st] = '_';
        }
        line2[row_size - 1] = ' ';
        if (last_move.row == i && last_move.col == BOARD_SIZE - 1) {
            line2[row_size - 1] = ']';
        }

        out << line2 << " " << 15 - i << endl;
        out << line << endl;
    }

    char column_no[row_size] = {0};
    int offset = 0;
    for (int i = 1; i <= BOARD_SIZE; i++) {
        offset += snprintf(column_no + offset, row_size - offset, "  %-2c", 'A' + i - 1);
    }
    out << column_no;
    puts(out.str().c_str());
}

void Game::show_result()
{
    if (is_over) {
        cout << "Game over! Result: ";
        switch (winner) {
            case COLOR_BLACK:
                cout << "BLACK \"X\" WINS";

                break;
            case COLOR_WHITE:
                cout << "WHITE \"O\" WINS";
                break;
            case COLOR_NONE:
                cout << "DRAW";
                break;
        }
        cout << endl;
    }
    else {
        cout << "Game is not over yet." << endl;
    }
}

void Game::check_is_over(int row, int col)
{
    winner = judge.do_judge(board, row, col);
    if (winner == COLOR_NONE) {
        if (steps == BOARD_SIZE * BOARD_SIZE - 1){
            is_over = true;
        }
        else {
            is_over = false;
        }
    }
    else {
        is_over = true;
    }
}

bool Game::is_legal_move(Color color, Position pos)
{
    return (pos.row >= 0 && pos.row < BOARD_SIZE && pos.col >= 0 && pos.col < BOARD_SIZE)
        && get(pos) == COLOR_NONE;
}

void Game::get_observation(Observation &obsv, Color pov)
{
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            auto c = get(i, j);
            if (c == pov) {
                obsv[0][i][j] = 1;
                obsv[1][i][j] = 0;
            }
            else if (c == -pov) {
                obsv[0][i][j] = 0;
                obsv[1][i][j] = 1;
            }
            else {
                obsv[0][i][j] = 0;
                obsv[1][i][j] = 0;
            }
        }
    }
}

