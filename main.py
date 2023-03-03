import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans
from math import ceil
from sklearn.metrics import silhouette_score


def make_dataset(cluster_size=3, num_samples=2000, show_dataset=False, is_convex=True):
    if is_convex:  # regular dataset
        X, labels = make_blobs(n_samples=num_samples, centers=cluster_size,
                               cluster_std=1.5, n_features=2, random_state=1)
    else:  # fun dataset
        X, labels = make_moons(n_samples=num_samples, shuffle=False, noise=0.1, random_state=1)
    if show_dataset:  # show the initial version of the dataset
        plt.scatter(X[:, 0], X[:, 1], s=3, color='y')
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title(("Blob" if is_convex else "Moon") + " Dataset")
        plt.show()

    return X


def setup_matplotlib():
    # settings for plotting - needed to show multiple plots
    size = 6
    plt.rc("font", size=size)
    plt.rc("axes", titlesize=size)
    plt.rc("axes", labelsize=size)
    plt.rc("xtick", labelsize=size)
    plt.rc("ytick", labelsize=size)
    plt.rc("figure", titlesize=size)


def draw_cluster_plot(axs, data, labels, centroids, iter_no, is_sklearn=False, is_show=False):
    if axs is None:
        _, axs = plt.subplots(1)
    for n_cluster in range(centroids.shape[0]):
        axs.scatter(data[labels == n_cluster, 0], data[labels == n_cluster, 1], s=3, label=n_cluster)
    axs.scatter(centroids[:, 0], centroids[:, 1], s=15, c="black", marker="*")
    axs.legend()
    axs.set_title(("Scikit Learn Results\n" if is_sklearn else "")
                  + "K = " + str(centroids.shape[0]) + " Iteration No = " + str(iter_no))
    axs.set(xlabel="X1", ylabel="X2")

    if is_show:
        plt.show()


def show_loss_plot(loss_list):
    plt.plot(range(1,len(loss_list) + 1), loss_list, marker="o", markersize=3)
    plt.xticks(range(0, len(loss_list) + 1, 3))
    plt.title("Objective Function")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.show()


def compute_initial_centroids(data, num_clusters):
    np.random.seed(1)  # for repeatable experiments
    # random initialization within the limits of the dataset
    centroids = np.random.uniform(np.min(data), np.max(data), (num_clusters, 2))
    return centroids


def compute_centroids(data, clustered, centroids):
    new_centroids = np.zeros((centroids.shape[0], 2))
    for cluster_no in range(centroids.shape[0]):
        # finds all values belongs to each cluster
        points_in_clusters = data[clustered == cluster_no]
        if points_in_clusters.shape[0] == 0:
            np.random.seed(1)  # for repeatable experiments
            # some cluster centers might not be chosen by any points when initialized randomly
            new_centroids[cluster_no] = np.random.uniform(np.min(data), np.max(data), (1, 2))
        else:
            # compute new centroid
            new_centroids[cluster_no] = np.mean(points_in_clusters, axis=0)

    return new_centroids


def assign_to_cluster(data, centroids):
    distances = np.zeros((centroids.shape[0], data.shape[0]))
    for i in range(centroids.shape[0]):
        # compute distance between each point and cluster centers
        distances[i, :] = norm(data - centroids[i], axis=1)
    # assign to the nearest cluster
    labels = np.argmin(distances, axis=0)

    return labels


def compute_loss(data, labels, centroids):
    euc_distance = np.zeros(data.shape[0])
    for i in range(centroids.shape[0]):
        euc_distance[labels == i] = norm(data[labels == i] - centroids[i], axis=1)
    return np.sum(np.square(euc_distance))


def fit_k_means(data, K, threshold=1e-3, show_plots=True, show_iters=[1, 2, 3]):

    # adjust plot space for iterations
    if show_plots:
        show_iters.insert(0, 0)  # show at least the initial
        subplt = plt.subplots(2, ceil((len(show_iters) + 1) / 2), squeeze=False)[1].flatten()
        if len(show_iters) % 2 == 0:
            subplt[-1].set_axis_off()

    # fit k-means clusters
    initial_centroids = compute_initial_centroids(data, K)
    losses = []
    centroids = initial_centroids
    n_iter = 0
    while True:
        old_centroids = centroids
        # find labels (assignment)
        clustered = assign_to_cluster(data, old_centroids)
        if n_iter in show_iters and show_plots:
            draw_cluster_plot(subplt[show_iters.index(n_iter)], data, clustered, centroids, n_iter, is_show=False)
        # new centroids based on new clusters
        centroids = compute_centroids(data, clustered, old_centroids)
        # compute objective to plot later
        loss = compute_loss(data, clustered, centroids)
        losses.append(loss)
        n_iter = n_iter + 1
        # stop in case of an early convergence
        if np.sum(norm(old_centroids - centroids)) < threshold:
            break

    if show_plots:
        # plot the last iteration
        draw_cluster_plot(subplt[len(show_iters)], data, clustered, centroids, n_iter, is_show=True)
        # plot objective function
        show_loss_plot(losses)

    return centroids, clustered, n_iter


def run_sklearn_clustering(data, K, threshold=1e-3, show_plots=True):
    # fit a model
    model = KMeans(n_clusters=K, tol=threshold, random_state=30).fit(data)
    # shows the final clustering
    if show_plots:
        draw_cluster_plot(None, data, model.labels_, model.cluster_centers_, model.n_iter_, is_sklearn=True, is_show=True)


def compute_silhouette_score(data, labels):
    # compute distance matrix
    distances = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        distances[i, i:] = norm(data[i:, :] - data[i], axis=1)
    distances = distances + distances.T

    # compute S matrix
    scores = np.zeros((data.shape[0], 1))
    for p_no, point in enumerate(data):
        p_distance = distances[p_no]
        assigned_cluster_no = labels[p_no]
        assigned_cluster_size = np.count_nonzero(labels == assigned_cluster_no)
        # penalize for single point clusters
        if assigned_cluster_size == 1:
            continue

        # compute a(i)
        points_in_dist = np.sum(p_distance[labels == assigned_cluster_no])
        a = points_in_dist / (assigned_cluster_size - 1)

        # compute b(i)
        points_out_dist_list = []
        for other_cluster_no in set(labels):
            if other_cluster_no == assigned_cluster_no:
                continue
            points_out_dist = np.mean(p_distance[labels == other_cluster_no])
            points_out_dist_list.append(points_out_dist)
        b = min(points_out_dist_list)

        # compute s(i)
        scores[p_no] = (b - a) / max(a, b)

    __silhouette_score = np.mean(scores)

    return __silhouette_score


def find_optimal_k(data, kmax=20):
    print("Finding optimal k ...")
    kmax = kmax if kmax < data.shape[0] - 1 else data.shape[0] - 1
    scores = []
    best_score = (0, -1)
    best_centroids, best_labels, best_total_iter = None, None, None
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k_iter in range(2, kmax + 1):
        centroids, labels, total_iter = fit_k_means(data=data, K=k_iter, show_plots=False)
        # uncomment to compare with the sklearn's implementation
        # k_score = silhouette_score(data, labels, metric='euclidean')
        k_score = compute_silhouette_score(data, labels)
        scores.append((k_iter, k_score))
        if k_score > best_score[1]:
            best_score = (k_iter, k_score)
            best_centroids, best_labels, best_total_iter = centroids, labels, total_iter

    print("Optimal k = " + str(best_centroids.shape[0]))

    fig, (axs1, axs2) = plt.subplots(1, 2)
    axs1.plot(*zip(*scores), marker="o", markersize=3)
    axs1.scatter(best_score[0], best_score[1], marker="*", c="red", zorder=5)
    axs1.annotate(text="max value " + str(best_score), xy=(best_score[0] + 0.5, best_score[1]))
    axs1.set_xticks(range(2, kmax + 1, 3), minor=False)
    axs1.set_title("Silhouette Score Function")
    axs1.set(xlabel="K", ylabel="Silhouette Score")

    draw_cluster_plot(axs2, data, best_labels, best_centroids, best_total_iter, is_sklearn=False, is_show=True)


if __name__ == '__main__':
    # set plot parameters
    setup_matplotlib()
    # make dataset - set is_convex to False for a more difficult dataset
    generated_data = make_dataset(cluster_size=3, num_samples=2000, show_dataset=True, is_convex=True)
    k = 3
    # run baseline algorithm - for show_iters[] 0 and the last iteration always included
    fit_k_means(data=generated_data, K=k, show_iters=[1, 2, 3], show_plots=True, threshold=1e-3)
    print("Clustering completed!\n")
    # run scikit-learn's algorithm
    run_sklearn_clustering(data=generated_data, K=k, show_plots=True, threshold=1e-3)
    print("Scikit-learn clustering completed!\n")
    # run Silhouette score algorithm
    find_optimal_k(data=generated_data)
