#include "test.hpp"

using namespace search;
using namespace std;


void test::run()
{
    Color current_color = COLOR_BLACK;

    Game game;
    game.graphic();
    while (!game.is_over) {
        Position pos;
        bool legal = false;
        while (!legal) {
            if (current_color == COLOR_BLACK) {
                printf("black player move: ");
            }
            else {
                printf("white player move: ");

            }
            char move_ic;
            int move_j;
            cin >> move_ic >> move_j;
            //scanf_s("%d", &move_j);
            pos = Position{move_ic - 'A', move_j - 1};
            legal = game.is_legal_move(current_color, pos);
            if (!legal) {
                printf("illegal input!\n");
                cin.clear();
                continue;
            }
            Game game_temp = game;
            game_temp.move(current_color, pos);
            game_temp.graphic();
            cout << "confirm? [y]/n" << endl;
            char ch;
            cin >> ch;
            if (ch == 'n') {
                game.graphic();
                legal = false;
            }
        }

        game.move(current_color, pos);

        game.graphic();
        current_color = -current_color;
    }
    game.show_result();
}
