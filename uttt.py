from ast import arg
import numpy as np
from copy import deepcopy as copy
from enum import Enum, unique
from numba import vectorize, guvectorize, int8

@unique
class FieldState(Enum):
    EMPTY = np.int8(0)
    FIRST = np.int8(1)
    SECOND = np.int8(2)
    TIE = np.int8(3)

    def __int__(self):
        return self.value

    def __str__(self):
        if(self is FieldState.FIRST):
            return "X"        
        if(self is FieldState.SECOND):
            return "O"
        return " "

@vectorize([int8(int8)], nopython=True)
def change_player(player):
    if player is FieldState.FIRST.value:
        return FieldState.SECOND.value
    if player is FieldState.SECOND.value:
        return FieldState.FIRST.value
    return player

# @guvectorize([(int8[:], int8[:])], "(n)->()", nopython=True)
# def get_victor(row, res):
#     if(np.all(row==row[0])):
#         res[:] = row[0] if row[0] != FieldState.TIE.value else FieldState.EMPTY.value
#     else: 
#         res[:] = FieldState.EMPTY.value

def add_guards(line, s):
    line = list(line)
    line.insert(6, s)
    line.insert(3, s)
    return line

class UltimateTicTacToe:
    def __init__(self) -> None:
        self.action_shape = (9,9)
        self.action_size = 81
        self.board_shape = self.action_shape
        self.allowed_field_size = self.action_shape[0]
        self.reset()

    def reset(self):
        self.board = np.zeros(self.board_shape, dtype = np.int8)
        self.allowed_field = 0
        self.allowed_mini_boards = np.zeros(self.action_shape[0], dtype=np.int8)
        self.player_turn = FieldState.FIRST.value
        self.done = False
        self.initial = True
        return copy(self.board), self.allowed_field, self.done, 0, "Reset"

    def __bool__(self) -> bool:
        return self.done

    def __repr__(self) -> str:
        return f"UltimateTicTacToe(done = {self.done})"

    def __str__(self) -> str:
        board = (row for row in self.board)
        board = (map(FieldState, row) for row in board)
        board = (map(str, row) for row in board)
        board = (list(row) for row in board)
        board = ("".join(add_guards(row, "|")) for row in board)
        board = add_guards(board, "".join(add_guards("-"*9, "+")))
        return "\n".join(str(row) for row in board)


    def clone(self):
        return copy(self)

    def getCanonicalState(self) -> bytes:
        return np.concatenate((np.concatenate(self.board), np.int8([self.allowed_field]), np.int8([self.player_turn]))).tobytes()

    def get_mini_field(self, i):
        return self.board[((i//3)*3):(i//3)*3+3,(i%3)*3:(i%3)*3+3]

    def calculate_mini_winner(self, mini_board) -> int:
        for row in mini_board:
            if ((row[0] == FieldState.FIRST.value or row[0] == FieldState.SECOND.value)
                and np.all(row==row[0])):
                return row[0]
        for row in mini_board.transpose():
            if ((row[0] == FieldState.FIRST.value or row[0] == FieldState.SECOND.value)
                and np.all(row==row[0])):
                return row[0]
        # winner = get_victor(self.board).max()
        # if(winner != 0):
        #     return winner
        # winner = get_victor(self.board.transpose()).max()
        # if(winner != 0):
        #     return winner
        if(mini_board[0][0] != FieldState.EMPTY.value
        and mini_board[0][0] != FieldState.TIE.value
        and mini_board[0][0] == mini_board[1][1] 
        and mini_board[1][1] == mini_board[2][2]):
            return mini_board[0][0]
        if(mini_board[2][0] != FieldState.EMPTY.value
        and mini_board[2][0] != FieldState.TIE.value
        and mini_board[2][0] == mini_board[1][1] 
        and mini_board[1][1] == mini_board[0][2]):
            return mini_board[2][0]
        if (mini_board == FieldState.EMPTY.value).any():
            return FieldState.EMPTY.value
        return FieldState.TIE.value

    def flip(self):
        self.player_turn = change_player(self.player_turn)
        self.allowed_mini_boards = change_player(self.allowed_mini_boards) #np.array(list(map(change_palyer, self.allowed_mini_boards)))
        self.board = change_player(self.board)
        return copy(self.board), self.allowed_field, self.done, 0, "Flip" 

    def play(self, *args):
        mini_board, field = None, None
        if(len(args) == 2):
            mini_board, field = args[0], args[1]
        else:
            a, b = args[0]//9, args[0]%9
            mini_board = (a//3)*3 + b//3
            field = (a%3)*3 + b%3
        if(self.done):
            return copy(self.board), self.allowed_field, self.done, -1, "Invalid move: Playing finished game"
        if(not self.initial):
            if(self.allowed_mini_boards[self.allowed_field] != FieldState.EMPTY.value and self.allowed_mini_boards[mini_board] != FieldState.EMPTY.value):
                self.done = True
                return copy(self.board), self.allowed_field, self.done, -1, "Invalid move: Mini board is already complete"
            if(self.allowed_mini_boards[self.allowed_field] == FieldState.EMPTY.value and mini_board != self.allowed_field):
                self.done = True
                return copy(self.board), self.allowed_field, self.done, -1, "Invalid move: Invalid mini board"
        else:
            self.initial = False

        action = (
            3*(mini_board//3) + field//3,
            3*(mini_board%3) + field%3
        )

        if(self.board[action] != FieldState.EMPTY.value):
            self.done = True
            return copy(self.board), self.allowed_field, self.done, -1, "Invalid move: Invalid field"

        self.board[action] = self.player_turn

        small_board = self.get_mini_field(mini_board)
        self.allowed_mini_boards[mini_board] = self.calculate_mini_winner(small_board)
        winner = self.calculate_mini_winner(self.allowed_mini_boards.reshape((3,3)))
        self.allowed_field = field

        if(winner == FieldState.EMPTY.value):
            self.player_turn = change_player(self.player_turn)
            return copy(self.board), self.allowed_field, self.done, 0, "Valid" 
        self.done = True
        if(winner == FieldState.TIE.value):
            return copy(self.board), self.allowed_field, self.done, 0, "Tie"
        if(winner == self.player_turn):
            return copy(self.board), self.allowed_field, self.done, 1, f"Player {self.player_turn} wins"
        return copy(self.board), self.allowed_field, self.done, -1, f"Player {self.player_turn} loses"
