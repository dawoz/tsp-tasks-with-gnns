from matplotlib import pyplot as plt

def plot_tour(x, tour, figsize=(5, 5), title=''):
    tour_r = tour.roll(-1, 0)
    plt.figure(figsize=figsize)
    plt.scatter(*x.t())
    plt.xlim(xmin=0, xmax=1)
    plt.ylim(ymin=0, ymax=1)
    plt.quiver(*x[tour].t(), *(x[tour_r] - x[tour]).t(), angles='xy', scale_units='xy', scale=1, units='dots', width=2)
    plt.title(title)

