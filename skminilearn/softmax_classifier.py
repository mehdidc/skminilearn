import numpy as np


class SoftmaxClassifier:

    def __init__(self, nb_epochs=1, batch_size=32, lr=1e-3, algo='sgd', random_state=None, verbose=1):
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.algo = algo
        self.random_state = random_state
        self.verbose = verbose
        self.nb_classes_ = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        assert len(X.shape) == 2
        assert len(y.shape) == 1
        assert y.dtype in (np.int32, np.int64)
        assert X.shape[0] == y.shape[0]

        self.nb_classes_ = len(np.unique(y))
        # initialize weights
        self.weights_ = rng.normal(size=(X.shape[1] + 1, self.nb_classes_))
        # accumulate gradients if adagrad is used
        self.weights_grad_accumulator_ = np.zeros_like(self.weights_)

        # training loop
        for epoch in range(self.nb_epochs):
            if self.verbose:
                print('Epoch {:05d}/{:05d}'.format(epoch + 1, self.nb_epochs))
            X, y = _shuffle(X, y, rng)
            for batch_start in range(0, X.shape[0], self.batch_size):
                # gather mini-batch
                batch_stop = batch_start + self.batch_size
                X_batch = X[batch_start:batch_stop]
                y_batch = y[batch_start:batch_stop]
                # compute loss
                loss = self._compute_loss(X_batch, y_batch)
                # backprop and update weights
                dweights = self._compute_gradients(X_batch, y_batch)
                self._update_weights(dweights)

                if self.verbose:
                    print(
                        '[{:04d}/{:04d}] Loss : {:4f}'.format(batch_start, len(X), loss))

    def predict_proba(self, X):
        X = _add_constant_column(X, value=1)
        scores = np.dot(X, self.weights_)
        probas = _softmax(scores)
        return probas

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def _compute_loss(self, X, y, average=True):
        X = _add_constant_column(X, value=1)
        scores = np.dot(X, self.weights_)
        probas = _softmax(scores)
        return -np.log(probas[np.arange(len(X)), y]).mean()

    def _compute_gradients(self, X, y):
        X = _add_constant_column(X, value=1)
        scores = np.dot(X, self.weights_)
        probas = _softmax(scores)
        dscores = probas.copy()
        dscores[np.arange(len(X)), y] -= 1
        dscores /= len(X)
        dweights = np.dot(X.T, dscores)
        return dweights

    def _update_weights(self, gradients):
        if self.algo == 'sgd':
            self._update_weights_sgd(gradients)
        elif self.algo == 'adagrad':
            self._update_weights_adagrad(gradients)
        else:
            raise ValueError('Algo "{}" not supported'.format(self.algo))

    def _update_weights_sgd(self, gradients):
        self.weights_ -= self.lr * gradients

    def _update_weights_adagrad(self, gradients):
        eps = 1e-10
        self.weights_grad_accumulator_ += gradients ** 2
        adjusted_grad = gradients / \
            (np.sqrt(self.weights_grad_accumulator_) + eps)
        self.weights_ -= self.lr * adjusted_grad


def _add_constant_column(X, value=1):
    col = np.ones((len(X), 1)) * value
    X = np.concatenate((col, X), axis=1)
    return X


def _softmax(scores):
    exp_score = np.exp(scores)
    norm = exp_score.sum(axis=1, keepdims=True)
    return exp_score / norm


def _shuffle(X, y, rng):
    assert len(X) == len(y)
    inds = np.arange(len(X))
    rng.shuffle(inds)
    return X[inds], y[inds]


def plot_decision_boundary(ax, clf, class_colors, min_val=-1, max_val=1, grid_resolution=100):
    xgrid = np.linspace(min_val, max_val, grid_resolution)
    ygrid = np.linspace(min_val, max_val, grid_resolution)
    xgrid, ygrid = np.meshgrid(xgrid, ygrid)
    grid = np.vstack((xgrid.flatten(), ygrid.flatten())).T
    assert grid.shape == (grid_resolution**2, 2)
    probas_grid = clf.predict_proba(grid)
    color_grid = np.dot(probas_grid, class_colors)
    color_grid = color_grid.reshape((grid_resolution, grid_resolution, 3))
    ax.imshow(color_grid)


def get_colors_from_cmap(cmap, nb_colors):
    result_colors = []
    color_indices = np.linspace(0, 255, nb_colors).astype(int)
    for color_index in color_indices:
        red, green, blue, opacity = cmap(color_index)
        result_colors.append((red, green, blue))
    return np.array(result_colors)


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    import matplotlib.pyplot as plt
    from matplotlib import cm
    # create dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_classes=4,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=0,
    )
    # fit classifier
    clf = SoftmaxClassifier(
        batch_size=1,
        nb_epochs=20,
        lr=0.1,
        algo='adagrad',
        verbose=0,
        random_state=0
    )
    clf.fit(X, y)
    acc = (clf.predict(X) == y).mean()
    print('Training accuracy : {:.2f}'.format(acc))
    # Plot decision boundary
    colors = get_colors_from_cmap(cm.viridis, clf.nb_classes_)
    min_val = X.min() - 1
    max_val = X.max() + 1
    grid_resolution = 100
    plot_decision_boundary(
        plt, clf, colors,
        min_val=min_val, max_val=max_val,
        grid_resolution=grid_resolution,
    )
    # rescale X to the scale of the decision boundary plot
    X_rescaled = grid_resolution * ((X - min_val) / (max_val - min_val))
    # the scatter have darker colors than the decision boundary
    scatter_colors = np.clip(colors - 0.2, 0, 1)
    plt.scatter(X_rescaled[:, 0], X_rescaled[:, 1], c=scatter_colors[y], s=10)
    plt.show()
