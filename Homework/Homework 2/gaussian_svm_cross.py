from matplotlib import pyplot as plt

xs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
os = [(-2, 0), (0, 2), (2, 0), (0, -2)]


def plot_data(title, show=True):
    # Plot the data
    for i in range(4):
        plt.plot(xs[i][0], xs[i][1], 'bx')
        plt.plot(os[i][0], os[i][1], 'ro')

    # Plot settings
    plt.title(title)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.xticks([-2, -1, 0, 1, 2])
    plt.yticks([-2, -1, 0, 1, 2])

    if show:
        plt.show()


plot_data("SVM")
