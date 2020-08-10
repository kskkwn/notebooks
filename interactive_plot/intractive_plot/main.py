import pylab as plt
import numpy as np

# onclick + subplots https://stackoverflow.com/questions/39351388/control-the-mouse-click-event-with-a-subplot-rather-than-a-figure-in-matplotlib


def get_data():
    return np.random.random(size=200).reshape(100, 2)


def update_plot(axes, data, knn_indexes):
    for i in range(1, 9):
        axes[i].cla()
        axes[i].scatter(data[:, 0], data[:, 1], alpha=0.3, marker=".", c="gray")
        axes[i].scatter(data[knn_indexes[i - 1]][0], data[knn_indexes[i - 1]][1], marker=".", c="C0")

    plt.draw()


def main():
    def onclick(event):
        try:
            index = axes.index(event.inaxes)
            print(index)

            inv_distances = (np.abs(data - query)[index])**(-1)
            weight = inv_distances / np.sum(inv_distances)
            print(weight)

            distances = np.sum((np.abs(data - query)) * weight, axis=1)
            knn_indexes = np.argsort(distances)[:k]
            print(knn_indexes)
            update_plot(axes, data, knn_indexes)

        except ValueError:
            pass

    data = get_data()
    query = np.array([0.5, 0.5])

    # initialize
    weight = np.ones(2)
    k = 8
    fig = plt.figure()
    axes = []

    for i in range(9):
        axes.append(fig.add_subplot(3, 3, i + 1))
        axes[-1].scatter(data[:, 0], data[:, 1], alpha=0.3, marker=".", c="gray")

    distances = np.sum((np.abs(data - query)) * weight, axis=1)
    knn_indexes = np.argsort(distances)[:k]
    axes[0].scatter(query[0], query[1], marker=".", c="C1")
    update_plot(axes, data, knn_indexes)

    fig.canvas.mpl_connect('button_press_event', onclick)  # start loop
    plt.show()


if __name__ == '__main__':
    main()
