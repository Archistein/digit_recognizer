from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt, QEvent
import torch.nn.functional as F
from model import CNN
import torch
import sys

WIN_WIDTH = 800
WIN_HEIGHT = 600

class MainWindow(QtWidgets.QMainWindow):

    CANVAS_SIZE = WIN_HEIGHT

    def __init__(self, WinWidth: int, WinHeight: int):
        super().__init__()

        self.model = CNN()
        self.model.load_state_dict(torch.load('params.pt'))
        self.model.eval()

        self.setFixedSize(WinWidth, WinHeight)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowMaximizeButtonHint)
        self.setWindowTitle('Handwritten digit recognizer')

        grid = QtWidgets.QHBoxLayout()
        vbox = QtWidgets.QVBoxLayout() 

        self.label = QtWidgets.QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.targets = [QtWidgets.QLabel() for _ in range(10)]

        for i in range(10):
            self.targets[i].setStyleSheet('font-size: 24px;')
            self.targets[i].setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.targets[i].setText(f'{i}:  0.00%')
            vbox.addWidget(self.targets[i])
        
        clearBtn = QtWidgets.QPushButton()
        clearBtn.setText('Clear (Ctrl+D)')
        clearBtn.clicked.connect(self.clear)
        vbox.addWidget(clearBtn)

        grid.setSpacing(0)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self.label)
        grid.addLayout(vbox)

        mainWidget = QtWidgets.QWidget()
        mainWidget.setLayout(grid)

        self.canvas = QtGui.QPixmap(self.CANVAS_SIZE, self.CANVAS_SIZE)
        self.canvas.fill(Qt.GlobalColor.white)

        self.prevPoint = None

        self.pen = QtGui.QPen()
        self.pen.setWidth(60)
        self.pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        self.label.setPixmap(self.canvas)
        self.setCentralWidget(mainWidget)

        QtGui.QShortcut(QtGui.QKeySequence('Ctrl+D'), self).activated.connect(self.clear)

    def mouseMoveEvent(self, event: QEvent) -> None:
        painter = QtGui.QPainter(self.canvas)

        pos = event.pos()

        painter.setPen(self.pen)   
        if self.prevPoint: painter.drawLine(self.prevPoint.x(), self.prevPoint.y(),
                                            pos.x(), pos.y())
        else: painter.drawPoint(pos.x(), pos.y())
        painter.end()

        self.label.setPixmap(self.canvas)

        self.prevPoint = pos

        # Real-time mode predictions
        # self.predict() 

    def mousePressEvent(self, event: QEvent) -> None:
        self.pen.setColor(Qt.GlobalColor.black)

        if event.button() == Qt.MouseButton.RightButton:
            self.pen.setColor(Qt.GlobalColor.white)


    def mouseReleaseEvent(self, event: QEvent) -> None:
        self.prevPoint = None

        # Release mode predictions
        self.predict() 

    def clear(self):
        self.canvas.fill(Qt.GlobalColor.white)
        self.label.setPixmap(self.canvas)

        for i in range(10):
            self.targets[i].setStyleSheet('font-size: 24px;')
            self.targets[i].setText(f'{i}: 0.00%')
    
    def predict(self):
        qimg = self.canvas.toImage().scaled(28, 28).convertToFormat(QtGui.QImage.Format.Format_Grayscale8) 
        qimg.invertPixels()
        x = torch.tensor(qimg.bits().asarray(28*28), dtype=torch.float32).view(1, 1, 28, 28) / 255
        
        with torch.no_grad():
            probs = F.softmax(self.model.forward(x), dim=1)

        for i in range(10):
            self.targets[i].setStyleSheet('font-size: 24px;')
            if i == probs.argmax().item(): 
                self.targets[i].setStyleSheet('''font-size: 30px;
                                                font-weight: bold; 
                                                text-decoration: underline;''')
            self.targets[i].setText(f'{i}: {probs[0, i].item() * 100:.2f}%')
       

def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(WIN_WIDTH, WIN_HEIGHT)
    window.show()
    app.exec()


if __name__ == '__main__':
    main()