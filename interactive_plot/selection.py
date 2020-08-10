import pickle
import pylab as plt
from glob import glob
from PIL import Image


class Selector(object):
    def __init__(self, data):
        self.data = data
        self.figure = plt.figure(figsize=(10, 5))

        self.plot_axes = [self.figure.add_subplot(1, 2, i + 1) for i in range(2)]
        self.indexes = []
        self.clicked_indexes = []
        for i, axis in enumerate(self.plot_axes):
            axis.imshow(data[i])
            self.indexes.append(i)

    def start(self):
        self.figure.canvas.mpl_connect('button_press_event', self._hook)
        plt.show()

    def _hook(self, event):
        index = self.plot_axes.index(event.inaxes)
        self.clicked_indexes.append(self.indexes[index])

        if (max(self.indexes) + 1) == len(self.data):
            return
        self.indexes[index] = max(self.indexes) + 1
        self.plot_axes[index].imshow(self.data[self.indexes[index]])
        plt.pause(0.001)

        with open("selection.pkl", "wb") as f:
            pickle.dump(self.clicked_indexes, f)


def main():
    data = [Image.open(filename) for filename in sorted(glob("./images/*"))]
    data = [img.resize((img.size[0] // 16, img.size[1] // 16)) for img in data]
    labeler = Selector(data)
    labeler.start()


if __name__ == '__main__':
    main()
