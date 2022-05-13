# This Python file uses the following encoding: utf-8
import sys
from enum import Enum, unique

from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtCore import QCoreApplication, Qt, QRectF
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QGraphicsScene, QGraphicsItem, QGraphicsObject
from PySide2.QtGui import QColor

from uttt import UltimateTicTacToe, FieldState, change_player
from model_handler import load_newest_model, load_best_model
from agent import Agent


GAP = 4
BOARD_SIZE = 600
MINIBOARD_SIZE = (BOARD_SIZE - 4 * GAP)/3
FIELD_SIZE = (MINIBOARD_SIZE - 4 * GAP)/3

@unique
class GameMode(Enum):
    NOT_CHOSEN = 0
    MVM = 1
    MVU = 2
    UVM = 3

env = UltimateTicTacToe()
game_mode = GameMode.NOT_CHOSEN
user = FieldState.EMPTY.value



class Field (QGraphicsItem):
    def __init__(self, parent):
        super().__init__()
        self.setParentItem(parent)
        self.state = FieldState.EMPTY.value
        self.index_in_miniboard = -1
        self.parent = parent

    def mousePressEvent(self, event):
        global env, game_mode, user
        super().mousePressEvent(event)
        if not env.done:
            if (game_mode == GameMode.UVM or game_mode == GameMode.MVU):
                if self.state == FieldState.EMPTY.value:
                    if env.player_turn == user:
                        _, info = (env.clone()).play(self.parent.index_in_board, self.index_in_miniboard)
                        if not info.startswith('Invalid move'):
                            self.state = env.player_turn
                            env.play(self.parent.index_in_board, self.index_in_miniboard)
                            if env.allowed_mini_boards[self.parent.index_in_board] == FieldState.FIRST.value:
                                parent.state = FieldState.FIRST.value
                            elif env.allowed_mini_boards[self.parent.index_in_board] == FieldState.SECOND.value:
                                parent.state = FieldState.SECOND.value
                            if env.done:
                                print("done")
                                self.update()
                                return
                            self.update()
                            self.parent.parent.model_play()

    def boundingRect(self):
        return QRectF(0, 0, FIELD_SIZE, FIELD_SIZE)

    def paint(self, painter, option, widget):
        painter.fillRect(self.boundingRect(), QColor(200, 200, 200))
        painter.setPen(QColor(0, 0, 0))
        if self.state == FieldState.FIRST.value:
            painter.drawLine(GAP,GAP,FIELD_SIZE - GAP, FIELD_SIZE - GAP)
            painter.drawLine(FIELD_SIZE - GAP, GAP, GAP, FIELD_SIZE - GAP)
        elif self.state == FieldState.SECOND.value:
            painter.drawEllipse(GAP,GAP,FIELD_SIZE - 2*GAP, FIELD_SIZE - 2*GAP)



class MiniBoard (QGraphicsItem):
    def __init__(self, parent):
        super().__init__()
        self.setParentItem(parent)
        self.fields = [
            [Field(self), Field(self), Field(self)],
            [Field(self), Field(self), Field(self)],
            [Field(self), Field(self), Field(self)]
        ]
        for i in range(3):
            for j in range(3):
                self.fields[i][j].index_in_minifield = i*3 + j
        self.index_in_board = -1
        self.parent = parent
        self.state = FieldState.EMPTY.value


    def boundingRect(self):
        return QRectF(0, 0, MINIBOARD_SIZE, MINIBOARD_SIZE)

    def paint(self, painter, option, widget):
        painter.fillRect(self.boundingRect(), QColor(200, 200, 200))
        painter.setPen(QColor(0, 0, 0))
        painter.drawLine(
            FIELD_SIZE + 3*GAP / 2, GAP,
            FIELD_SIZE + 3*GAP / 2, MINIBOARD_SIZE - GAP
        )
        painter.drawLine(
            2*FIELD_SIZE + 5*GAP / 2, GAP,
            2*FIELD_SIZE + 5*GAP / 2, MINIBOARD_SIZE - GAP
        )
        painter.drawLine(
            GAP, FIELD_SIZE + 3*GAP / 2,
            MINIBOARD_SIZE - GAP, FIELD_SIZE + 3*GAP / 2
        )
        painter.drawLine(
            GAP, 2*FIELD_SIZE + 5*GAP / 2,
            MINIBOARD_SIZE - GAP, 2*FIELD_SIZE + 5*GAP / 2
        )
        for i in range(3):
            for j in range(3):
                self.fields[i][j].setPos(
                    GAP + FIELD_SIZE*i + GAP*i,
                    GAP + FIELD_SIZE*j + GAP*j
                )
        if self.state == FieldState.FIRST.value:
            painter.drawLine(GAP,GAP,MINIBOARD_SIZE - GAP, MINIBOARD_SIZE - GAP)
            painter.drawLine(MINIBOARD_SIZE - GAP, GAP, GAP, MINIBOARD_SIZE - GAP)
        elif self.state == FieldState.SECOND.value:
            painter.drawEllipse(GAP,GAP,MINIBOARD_SIZE - 2*GAP, MINIBOARD_SIZE - 2*GAP)


class Board (QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.mini_boards = [
            [MiniBoard(self), MiniBoard(self), MiniBoard(self)],
            [MiniBoard(self), MiniBoard(self), MiniBoard(self)],
            [MiniBoard(self), MiniBoard(self), MiniBoard(self)]
        ]
        for i in range(3):
            for j in range(3):
                self.mini_boards[i][j].index_in_board = i*3 + j

    def boundingRect(self):
        return QRectF(0, 0, BOARD_SIZE, BOARD_SIZE)

    def model_play(self):
        global env, game_mode, user, agent
        agent.play_action()
        action = agent.action
        a, b = action//9, action%9
        mb = (a//3)*3 + b//3
        f = (a%3)*3 + b%3
        turn = change_player(user)
        self.mini_boards[mb/3,mb%3].fields[f/3,f%3].state = turn
        if env.allowed_mini_boards[mb] == FieldState.FIRST.value:
            self.mini_boards[mb/3,mb%3].state = FieldState.FIRST.value
        elif env.allowed_mini_boards[mb] == FieldState.SECOND.value:
            self.mini_boards[mb/3,mb%3].state = FieldState.SECOND.value
        if env.done:
            print("done")
        self.mini_boards[mb/3,mb%3].fields[f/3,f%3].update()

    def paint(self, painter, option, widget):
        painter.fillRect(self.boundingRect(), QColor(200, 200, 200))
        painter.setPen(QColor(0, 0, 0))
        painter.drawLine(
            MINIBOARD_SIZE + 3*GAP / 2, GAP,
            MINIBOARD_SIZE + 3*GAP / 2, BOARD_SIZE - GAP
        )
        painter.drawLine(
            2*MINIBOARD_SIZE + 5*GAP / 2, GAP,
            2*MINIBOARD_SIZE + 5*GAP / 2, BOARD_SIZE - GAP
        )
        painter.drawLine(
            GAP, MINIBOARD_SIZE + 3*GAP / 2,
            BOARD_SIZE - GAP, MINIBOARD_SIZE + 3*GAP / 2
        )
        painter.drawLine(
            GAP, 2*MINIBOARD_SIZE + 5*GAP / 2,
            BOARD_SIZE - GAP, 2*MINIBOARD_SIZE + 5*GAP / 2
        )
        for i in range(3):
            for j in range(3):
                self.mini_boards[i][j].setPos(
                    GAP + MINIBOARD_SIZE*i + GAP*i,
                    GAP + MINIBOARD_SIZE*j + GAP*j
                )


class mainGraphicsScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.board = Board()
        self.addItem(self.board)


def Model_Vs_Model():
    global env, game_mode, user
    game_mode = GameMode.MVM
    env.reset()
    newest = load_newest_model()
    best = load_best_model()


def Model_Vs_User():
    global env, game_mode, user
    env.reset()
    game_mode = GameMode.MVU
    user = FieldState.SECOND.value
    model = load_best_model()
    agent = Agent(model, env)


def User_Vs_Model():
    global env, game_mode, user, agent
    env.reset()
    game_mode = GameMode.UVM
    user = FieldState.FIRST.value
    model = load_best_model()
    agent = Agent(model, env)



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = QUiLoader().load("form.ui")
        self.scene = mainGraphicsScene()
        self.ui.mainGV.setScene(self.scene)
        self.ui.action_model_vs_model.triggered.connect(Model_Vs_Model)
        self.ui.action_model_X_vs_user_O.triggered.connect(Model_Vs_User)
        self.ui.action_user_X_vs_model_O.triggered.connect(User_Vs_Model)
        self.setMouseTracking(True)


if __name__ == "__main__":
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication([])
    widget = MainWindow()
    widget.ui.show()
    sys.exit(app.exec_())
