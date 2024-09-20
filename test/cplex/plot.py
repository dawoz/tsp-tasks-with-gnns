import matplotlib.pyplot as plt


def plot_tsp(X, tour=None, dpi=80):
    plt.figure(dpi=dpi)
    
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], i, ha='center', va='center', fontsize=10, color='black', zorder=30)
        plt.scatter(X[i, 0], X[i, 1], c='orange', linewidths=6, zorder=10)    

    if tour is not None:
        for i, j in tour:
            plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'k-')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')