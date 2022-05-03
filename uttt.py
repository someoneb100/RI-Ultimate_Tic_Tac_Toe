from ast import arg
import numpy as np
from copy import deepcopy as copy
from enum import Enum, unique

@unique
class FieldState(Enum):
    EMPTY = 0
    FIRST = 1
    SECOND = 2
    TIE = 3

    def __int__(self):
        return self.value

    def __str__(self):
        if(self is FieldState.FIRST):
            return "X"        
        if(self is FieldState.SECOND):
            return "O"
        return " "

def add_guards(line, s):
    line = list(line)
    line.insert(6, s)
    line.insert(3, s)
    return line

class UltimateTicTacToe:
    def __init__(self) -> None:
        self.action_shape = (9,9)
        self.board_shape = self.action_shape
        self.allowed_field_size = self.action_shape[0]
        self.reset()

    def reset(self):
        self.board = np.zeros(self.board_shape, dtype = np.float)
        self.allowed_field = 0
        self.allowed_mini_boards = np.zeros(self.action_shape[0], dtype=np.int)
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

    def calculate_mini_winner(self, mini_board) -> int:
        for row in mini_board:
            if (row[0] != FieldState.EMPTY.value
                and row[0] != FieldState.TIE.value
                and (row == row[0]).all()):
                return row[0]
        for col in mini_board.transpose():
            if (row[0] != FieldState.EMPTY.value
                and row[0] != FieldState.TIE.value
                and (col == col[0]).all()):
                return col[0]
        if(mini_board[0][0] != FieldState.EMPTY.value
        and mini_board[0][0] != FieldState.TIE.value
        and mini_board[0][0] == mini_board[1][1] 
        and mini_board[1][1] == mini_board[2][2]):
            return mini_board[0][0]
        if(mini_board[2][0] != FieldState.EMPTY.value
        and mini_board[2][0] != FieldState.TIE.value
        and mini_board[2][0] == mini_board[1][1] 
        and mini_board[1][1] == mini_board[2][0]):
            return mini_board[2][0]
        if (mini_board == FieldState.EMPTY.value).any().any():
            return FieldState.EMPTY.value
        return FieldState.EMPTY.value

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
            if(self.allowed_mini_boards[self.allowed_field] == FieldState.EMPTY.value and mini_board != self.allowed_field):
                self.done = True
                return copy(self.board), self.allowed_field, self.done, -1, "Invalid move: Invalid mini board"
            if(self.allowed_mini_boards[mini_board] != FieldState.EMPTY.value):
                self.done = True
                return copy(self.board), self.allowed_field, self.done, -1, "Invalid move: Mini board already full"
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

        small_board = np.zeros((3,3), dtype=np.int)
        m = mini_board//3, mini_board%3
        for i in range(3):
            for j in range(3):
                small_board[i][j] = self.board[3*m[0] + i][3*m[1] + j]
        self.allowed_mini_boards[mini_board] = self.calculate_mini_winner(small_board)
        winner = self.calculate_mini_winner(self.allowed_mini_boards.reshape((3,3)))
        self.allowed_field = field

        if(winner == FieldState.EMPTY.value):
            self.player_turn = FieldState.SECOND.value if self.player_turn is FieldState.FIRST.value else FieldState.FIRST.value
            return copy(self.board), self.allowed_field, self.done, 0, "Valid" 
        self.done = True
        if(winner == FieldState.TIE.value):
            return copy(self.board), self.allowed_field, self.done, 0, "Tie"
        if(winner == self.player_turn):
            return copy(self.board), self.allowed_field, self.done, 1, f"Player {self.player_turn} wins"
        return copy(self.board), self.allowed_field, self.done, -1, f"Player {self.player_turn} loses"
