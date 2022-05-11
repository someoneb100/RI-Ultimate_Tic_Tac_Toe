import numpy as np
from copy import deepcopy as copy
from enum import Enum, unique
from numba import vectorize, njit, int8, boolean

@unique
class FieldState(Enum):
    EMPTY = np.int8(0)
    FIRST = np.int8(1)
    SECOND = np.int8(-1)
    TIE = np.int8(2)

    def __int__(self) -> int:
        return self.value

    def __str__(self) -> str:
        if(self is FieldState.FIRST):
            return "X"        
        if(self is FieldState.SECOND):
            return "O"
        return " "

@vectorize([int8(int8)], nopython=True)
def change_player(player: int) -> int:
    if player == FieldState.FIRST.value:
        return FieldState.SECOND.value
    if player == FieldState.SECOND.value:
        return FieldState.FIRST.value
    return player

@njit([int8(int8[:,:])])
def calculate_mini_winner(mini_board) -> int:
    for i in range(3):
        if ((mini_board[i][0] == FieldState.FIRST.value or mini_board[i][0] == FieldState.SECOND.value)
        and np.all(mini_board[i]==mini_board[i][0])):
            return mini_board[i][0]

    mini_board = mini_board.transpose()
    for i in range(3):
        if ((mini_board[i][0] == FieldState.FIRST.value or mini_board[i][0] == FieldState.SECOND.value)
        and np.all(mini_board[i]==mini_board[i][0])):
            return mini_board[i][0]
            
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

@njit([int8[:](boolean, int8)])
def categorical_allowed_field(initial: bool, i: int):
    res = np.zeros(9, dtype=np.int8)
    if not initial:
        res[i] = 1
    return res

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

    def reset(self) -> None:
        self.board = np.zeros(self.board_shape, dtype = np.int8)
        self.allowed_field = 0
        self.allowed_mini_boards = np.zeros(self.action_shape[0], dtype=np.int8)
        self.player_turn = FieldState.FIRST.value
        self.done = False
        self.initial = True

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

    def get_categorical_allowed_field(self) -> np.ndarray:
        return categorical_allowed_field(self.initial, self.allowed_field)

    def clone(self) -> "UltimateTicTacToe":
        return copy(self)

    def getCanonicalState(self) -> bytes:
        return np.concatenate((np.concatenate(self.board), np.int8([self.allowed_field]), np.int8([self.player_turn]))).tobytes()

    def get_mini_field(self, i: int) -> np.ndarray:
        return self.board[((i//3)*3):(i//3)*3+3,(i%3)*3:(i%3)*3+3]

    def flip(self) -> None:
        self.player_turn = change_player(self.player_turn)
        self.allowed_mini_boards = change_player(self.allowed_mini_boards) #np.array(list(map(change_palyer, self.allowed_mini_boards)))
        self.board = change_player(self.board)

    def play(self, *args) -> "tuple[int,str]":
        mini_board, field = None, None
        if(len(args) == 2):
            mini_board, field = args[0], args[1]
        else:
            a, b = args[0]//9, args[0]%9
            mini_board = (a//3)*3 + b//3
            field = (a%3)*3 + b%3
        if(self.done):
            return -1, "Invalid move: Playing finished game"
        if(not self.initial):
            if(self.allowed_mini_boards[self.allowed_field] != FieldState.EMPTY.value and self.allowed_mini_boards[mini_board] != FieldState.EMPTY.value):
                self.done = True
                return -1, "Invalid move: Mini board is already complete"
            if(self.allowed_mini_boards[self.allowed_field] == FieldState.EMPTY.value and mini_board != self.allowed_field):
                self.done = True
                return -1, "Invalid move: Invalid mini board"
        else:
            self.initial = False

        action = (
            3*(mini_board//3) + field//3,
            3*(mini_board%3) + field%3
        )

        if(self.board[action] != FieldState.EMPTY.value):
            self.done = True
            return -1, "Invalid move: Invalid field"

        self.board[action] = self.player_turn

        small_board = self.get_mini_field(mini_board)
        self.allowed_mini_boards[mini_board] = calculate_mini_winner(small_board)
        winner = calculate_mini_winner(self.allowed_mini_boards.reshape((3,3)))
        self.allowed_field = field

        if(winner == FieldState.EMPTY.value):
            self.player_turn = change_player(self.player_turn)
            return 0, "Valid" 
        self.done = True
        if(winner == FieldState.TIE.value):
            return 0, "Tie"
        if(winner == self.player_turn):
            return 1, f"Player {self.player_turn} wins"
        return -1, f"Player {self.player_turn} loses"
