import pickle
import pylab as plt
from glob import glob
from PIL import Image


class Labeler(object):
    def __init__(self, data):
        self.data = data
        self.labels = [-1 for _ in data]
        self.ind = 0

    def start(self):
        self.figure = plt.figure(figsize=(5, 5))
        self.figure.canvas.mpl_connect("key_press_event", self._hook)
        self.axis = self.figure.add_subplot(1, 1, 1)
        self.axis.imshow(self.data[self.ind])
        plt.show()

    def _hook(self, event):
        print(event.key)
        if event.key == "right":
            self.ind = min(self.ind + 1, len(self.data) - 1)
        elif event.key == "left":
            self.ind = max(self.ind - 1, 0)
        elif event.key == "q":
            with open("labels.pkl", "wb") as f:
                pickle.dump(self.labels, f)
        else:
            self.labels[self.ind] = event.key
            self.ind = min(self.ind + 1, len(self.data) - 1)
        self.axis.imshow(self.data[self.ind])
        plt.pause(0.001)


def main():
    data = [Image.open(filename) for filename in sorted(glob("./images/*"))]
    data = [img.resize((img.size[0] // 16, img.size[1] // 16)) for img in data]
    labeler = Labeler(data)
    labeler.start()


if __name__ == '__main__':
    main()
