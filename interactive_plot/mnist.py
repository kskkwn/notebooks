from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import pylab as plt
import numpy as np


class InteractiveKNNPlotter(object):
    def __init__(self, query, data, nb_cols=3, nb_rows=3):
        self.k = 8
        self.figure = plt.figure(figsize=(5, 5))

        self.data = data.reshape(-1, 28, 28)

        self.query_axis = self.figure.add_subplot(nb_rows, nb_cols, 1)
        self.query_axis.imshow(query.reshape(28, 28))

        self.plot_axes = [self.figure.add_subplot(nb_rows, nb_cols, i + 2) for i in range(8)]

        pca = PCA(n_components=0.98)
        self.pcaed_data = pca.fit_transform(data)
        self.weight = np.ones(self.pcaed_data.shape[1]) / self.pcaed_data.shape[1]
        self.pcaed_query = pca.transform(query[None])

        self._update_distances_and_knn()
        self._update_plot()
        self.figure.canvas.mpl_connect('button_press_event', self._onclick_loop)  # start loop

        plt.show()

    def _update_distances_and_knn(self):
        self.distances = np.sum(np.abs(self.pcaed_data - self.pcaed_query) * self.weight, axis=1)
        self.knn_indices = np.argsort(self.distances)[:self.k]

    def _update_plot(self):
        for i, axis in enumerate(self.plot_axes):
            axis.imshow(self.data[self.knn_indices[i]])
        plt.draw()

    def _update_weight(self, index):
        temp_weight = np.abs(self.pcaed_data[self.knn_indices[index]] - self.pcaed_query)**(-1)
        temp_weight /= np.sum(temp_weight)
        temp_weight = temp_weight[0]

        self.weight += temp_weight * 0.1
        self.weight /= np.sum(self.weight)

    def _onclick_loop(self, event):
        # try:
        index = self.plot_axes.index(event.inaxes)
        print(index)
        self._update_weight(index)
        self._update_distances_and_knn()
        self._update_plot()

        # except ValueError:
        #     pass


if __name__ == '__main__':
    data, _ = fetch_openml('mnist_784', version=1, return_X_y=True)
    query = data[0]
    data = data[1:]

    plotter = InteractiveKNNPlotter(query, data)
