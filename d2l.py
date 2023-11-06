import time
import random
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from IPython import display

from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt
import numpy as np


def use_svg_display():
    """use the svg format to display a plot in Jupyter"""
    backend_inline.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):
    """set the figure size for matplotlib"""
    # use_svg_display()
    backend_inline.set_matplotlib_formats("svg")
    # plt.rcParams change the default properties of chart
    plt.rcParams["figure.figsize"] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(
    X,
    Y=None,
    xlabel=None,
    ylabel=None,
    legend=None,
    xlim=None,
    ylim=None,
    xscale="linear",
    yscale="linear",
    fmts=("-", "m--", "g-.", "r:"),
    figsize=(3.5, 2.5),
    axes=None,
):
    if legend is None:
        legend = []
    set_figsize(figsize)
    # gca() indicated as get current axes
    axes = axes if axes else plt.gca()

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (
            hasattr(X, "ndim")
            and X.ndim == 1
            or isinstance(X, list)
            and not hasattr(X[0], "__len__")
        )

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None:
        axes = plt.gca()
    # clear axes(the active axes in the current figure)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)  # if X has no data
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    # plt.show()


class Timer:
    """Record multiple running times"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        """time.time() return the current time"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        """return the accumulated time"""
        return np.array(self.times).cumsum().tolist()


def synthetic_data(w, b, num_examples):
    """Generate y =Xw + b + noise"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    # plus noise. y'shape is num_examples*1
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def linreg(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    # y_hat indicated as estimated value
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    # after torch.no_grad(). the all calcuated tensor' required_grad is set to false
    with torch.no_grad():
        for param in params:
            # param.grad calcuate the sum of a batch_size
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator"""
    # the asterisk in front of data_arrays indicated unpacking data structure like(X,Y)
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def get_fasion_mnnnist_labels(labels):
    """Return text labels for the Fashion-MNIST dataset"""
    text_labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    # plt.subplot() specify the partition and location to draw
    print(1)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    # axes is n*m tuple. axes=axes.flatten() -> axes is 1*nm tuple
    print(2)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # tensor of picture
            ax.imshow(img.numpy())
        else:
            # PIL picture
            plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()
    # return axes


def get_dataloader_workers():
    # use 4 processes to read the data
    return 4


def load_data_fasion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it
    into memory"""
    # one for training and testing
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True
    )
    return (
        data.DataLoader(
            mnist_train,
            batch_size,
            shuffle=True,
            num_workers=get_dataloader_workers(),
        ),
        data.DataLoader(
            mnist_test,
            batch_size,
            shuffle=False,
            num_workers=get_dataloader_workers(),
        ),
    )


def accuracy(y_hat, y):
    """y_hat prediction distribution"""
    """y tags"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n  # data equal to a list has n zero elements

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):
    """calcuate the accuracy of model on specified dataset"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # set the model to evaluation mode
    metric = Accumulator(2)  # correct predictions and total number of predictions
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    """train model for one epoch"""
    # set the mode to training mode
    if isinstance(net, torch.nn.Module):
        net.train()

    # sum of training loss, sum of training accuracy, num of samples
    metric = Accumulator(3)

    for X, y in train_iter:
        # calcuate the parameters of grad and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # use PyTorch's builtin optimizer and loss function
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # use a custom optimizer and loss function
            l.sum.backward()
            updater(X.shape[0])
        metric.add(float(l.sum), accuracy(y_hat, y), y.numel())

        # return training loss and training accuracy
        return metric[0] / metric[2], metric[1] / metric[2]


class Animator:
    """draw data in animation"""

    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        legend=None,
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
        fmts=("-", "m--", "g-.", "r:"),
        nrows=1,
        ncols=1,
        figsize=(3.5, 2.5),
    ):
        # draw multiple lines incremently
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [
                self.axes,
            ]

        # use lambda function to capture arguments
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend
        )
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # add multiple data point to a chart
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]

        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()

        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
