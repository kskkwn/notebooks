import pickle
import pylab as plt
from glob import glob
from PIL import Image
import matplotlib.patches as patches


class Detector(object):
    def __init__(self, data):
        self.data = data
        self.rects = [[] for _ in data]
        self.objects = [[] for _ in data]
        self.ind = 0

        self.dragging = False

    def start(self):
        self.figure = plt.figure(figsize=(5, 5))
        self.figure.canvas.mpl_connect("key_press_event", self._key_hook)
        self.figure.canvas.mpl_connect("button_press_event", self._press_hook)
        self.figure.canvas.mpl_connect("button_release_event", self._release_hook)
        self.figure.canvas.mpl_connect("motion_notify_event", self._motion_hook)

        self.axis = self.figure.add_subplot(1, 1, 1)
        self.axis.imshow(self.data[self.ind])

        plt.show()

    def _key_hook(self, event):
        if event.key == "right":
            self.axis.cla()
            self.ind = min(self.ind + 1, len(self.data) - 1)
            for r in self.rects[self.ind]:
                self.axis.add_patch(r)

        elif event.key == "left":
            self.axis.cla()
            self.ind = max(self.ind - 1, 0)
            for r in self.rects[self.ind]:
                self.axis.add_patch(r)
        elif event.key == "q":
            with open("objects.pkl", "wb") as f:
                pickle.dump(self.objects, f)
            plt.close(self.figure)
        elif event.key == "c":
            if len(self.rects[self.ind]) > 0:
                self.rects[self.ind][-1].remove()
                del self.rects[self.ind][-1]
        else:
            self.ind = min(self.ind + 1, len(self.data) - 1)

        self.axis.imshow(self.data[self.ind])
        plt.pause(0.001)

    def _press_hook(self, event):
        if event.inaxes != self.axis:
            return
        if event.button == 1:
            self.start_xy = (event.xdata, event.ydata)

        r = patches.Rectangle(xy=self.start_xy,
                              width=abs(self.start_xy[0] - event.xdata),
                              height=abs(self.start_xy[1] - event.ydata),
                              fill=False)
        self.axis.add_patch(r)
        self.rects[self.ind].append(r)

        self.dragging = True
        plt.pause(0.001)

    def _motion_hook(self, event):
        if len(self.rects[self.ind]) == 0:
            return
        if not self.dragging:
            return

        if event.xdata is not None:
            self.rects[self.ind][-1].set_x(min(self.start_xy[0], event.xdata))
            self.rects[self.ind][-1].set_width(abs(self.start_xy[0] - event.xdata))
        if event.ydata is not None:
            self.rects[self.ind][-1].set_y(min(self.start_xy[1], event.ydata))
            self.rects[self.ind][-1].set_height(abs(self.start_xy[1] - event.ydata))
        plt.pause(0.001)

    def _release_hook(self, event):
        x = self.rects[self.ind][-1].get_x()
        y = self.rects[self.ind][-1].get_y()
        w = self.rects[self.ind][-1].get_width()
        h = self.rects[self.ind][-1].get_height()

        self.objects[self.ind].append([x, y, w, h])
        self.dragging = False

        print(self.objects)


def main():
    data = [Image.open(filename) for filename in sorted(glob("./images/*"))]
    data = [img.resize((img.size[0] // 16, img.size[1] // 16)) for img in data]
    detector = Detector(data)
    detector.start()


if __name__ == '__main__':
    main()
