import numpy as np
import pylab as plt


class OnClickPlotter(object):
    def __init__(self, figsize=(5, 5), nb_cols=3, nb_rows=3):
        self.figure = plt.figure(figsize=figsize)
        self.plot_axes = [self.figure.add_subplot(nb_rows, nb_cols, i + 1)
                          for i in range(nb_cols * nb_rows)]

        self.figure.canvas.mpl_connect('button_press_event', self._on_click)

        plt.show()

    def _on_click(self, event):
        print("(x,y) in the plot:", event.xdata, event.ydata)
        print("(x,y) in the fig:", event.x, event.y)
        print("axis ind:", self.plot_axes.index(event.inaxes))
        print("left or right", event.button)


if __name__ == '__main__':
    plotter = OnClickPlotter()
