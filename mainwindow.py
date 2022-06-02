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

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


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



class Field (QGraphicsItem):
    def __init__(self, parent, index):
        super().__init__()
        self.setParentItem(parent)
        self.state = FieldState.EMPTY.value
        self.index_in_miniboard = index
        self.parent = parent

    def mousePressEvent(self, event):
        global widget
        env = self.parent.parent.env
        game_mode = self.parent.parent.game_mode
        user = self.parent.parent.user
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
                                self.parent.setState(FieldState.FIRST.value)
                            elif env.allowed_mini_boards[self.parent.index_in_board] == FieldState.SECOND.value:
                                self.parent.setState(FieldState.SECOND.value)
                            if env.done:
                                print("done")
                                widget.ui.mainGV.viewport().repaint()
                                return
                            widget.ui.mainGV.viewport().repaint()
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
    def __init__(self, parent, index):
        super().__init__()
        self.setParentItem(parent)
        self.fields = []
        for i in range(9):
            self.fields.append(Field(self, i))
        self.index_in_board = index
        self.parent = parent
        self.state = FieldState.EMPTY.value

    def setState(self, state):
        self.state = state
        for i in range(9):
            self.fields[i].hide()

    def boundingRect(self):
        return QRectF(0, 0, MINIBOARD_SIZE, MINIBOARD_SIZE)

    def paint(self, painter, option, widget):
        painter.setPen(QColor(0, 0, 0))
        if self.state == FieldState.EMPTY.value:
            painter.fillRect(self.boundingRect(), QColor(200, 200, 200))
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
                    self.fields[3*i + j].setPos(
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
        self.mini_boards = []
        for i in range(9):
            self.mini_boards.append(MiniBoard(self, i))
        self.env = UltimateTicTacToe()
        self.game_mode = GameMode.NOT_CHOSEN
        self.user = FieldState.EMPTY.value
        self.agent = None

    def setState(self, state):
        self.state = state
        for i in range(9):
            self.mini_boards[i].hide()

    def boundingRect(self):
        return QRectF(0, 0, BOARD_SIZE, BOARD_SIZE)

    def model_play(self):
        global widget
        if self.game_mode == GameMode.MVM:
            turn = self.env.player_turn
        else:
            turn = change_player(self.user)
        self.agent.play_action()
        action = self.agent.action
        a, b = action//9, action % 9
        mb = (a//3)*3 + b//3
        f = (a % 3)*3 + b % 3
        self.mini_boards[mb].fields[f].state = turn
        if self.env.allowed_mini_boards[mb] == FieldState.FIRST.value:
            self.mini_boards[mb].setState(FieldState.FIRST.value)
        elif self.env.allowed_mini_boards[mb] == FieldState.SECOND.value:
            self.mini_boards[mb].setState(FieldState.SECOND.value)
        if self.env.done:
            print("done")
        widget.ui.mainGV.viewport().repaint()


    def Model_Vs_Model(self):
        self.game_mode = GameMode.MVM
        self.env.reset()
        newest = load_newest_model()
        best = load_best_model()
        newest_agent = Agent(newest,self.env)
        best_agent = Agent(best,self.env)
        while(True):
            self.agent = newest_agent
            self.model_play()
            if self.env.done:
                print("done")
                break
            self.agent = best_agent
            self.model_play()
            if self.env.done:
                print("done")
                break

    def Model_Vs_User(self):
        self.env.reset()
        self.game_mode = GameMode.MVU
        self.user = FieldState.SECOND.value
        model = load_best_model()
        self.agent = Agent(model, self.env)
        self.model_play()


    def User_Vs_Model(self):
        self.env.reset()
        self.game_mode = GameMode.UVM
        self.user = FieldState.FIRST.value
        model = load_best_model()
        self.agent = Agent(model, self.env)

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
                self.mini_boards[3*i + j].setPos(
                    GAP + MINIBOARD_SIZE*i + GAP*i,
                    GAP + MINIBOARD_SIZE*j + GAP*j
                )


class mainGraphicsScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.board = Board()
        self.addItem(self.board)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = QUiLoader().load("form.ui")
        self.scene = mainGraphicsScene()
        self.ui.mainGV.setScene(self.scene)
        self.ui.action_model_vs_model.triggered.connect(self.scene.board.Model_Vs_Model)
        self.ui.action_model_X_vs_user_O.triggered.connect(self.scene.board.Model_Vs_User)
        self.ui.action_user_X_vs_model_O.triggered.connect(self.scene.board.User_Vs_Model)
        self.setMouseTracking(True)


if __name__ == "__main__":
    global widget
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication([])
    widget = MainWindow()
    widget.ui.show()
    sys.exit(app.exec_())
