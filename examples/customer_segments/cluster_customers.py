"""Customer segmentation with cuhc and scikit-learn on the Kaggle "Mall Customers" data.

The data is Mall_Customers.csv from the Kaggle dataset
vjchoudhary7/customer-segmentation-tutorial-in-python (200 customers). By default
the script reads ~/Downloads/Mall_Customers.csv; use --data for another path.

The script clusters the customers two ways, with cuhc and with scikit-learn:
  1. Ward linkage, cut into 5 segments (AgglomerativeClustering), with its dendrogram,
  2. HDBSCAN, which finds dense groups and marks the remaining points as noise.

The figure has one row for cuhc and one for scikit-learn. Each library numbers its
clusters in its own way, so the scikit-learn clusters are renumbered to match the
cuhc clusters they overlap most. Then one segment has the same colour in both rows,
and any difference between the two libraries shows as a different colour.

It clusters on annual income and spending score, the two features that separate
the customer segments. (CustomerID is a row number, so it is not a feature.)

Usage:
    python cluster_customers.py                       # show the figure
    python cluster_customers.py --save out.png        # write it to a file instead
    python cluster_customers.py --data path/to/Mall_Customers.csv
"""

import argparse
import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from scipy.cluster.hierarchy import dendrogram
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import HDBSCAN as SkHDBSCAN
from sklearn.cluster import AgglomerativeClustering as SkAgglomerative
from sklearn.metrics import adjusted_rand_score

import cuhc

DATA = pathlib.Path.home() / "Downloads" / "Mall_Customers.csv"
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
N_SEGMENTS = 5
# HDBSCAN settings. With 200 customers the outer segments are sparse, so the core
# distance uses fewer neighbours (min_samples) than the smallest cluster size.
MIN_CLUSTER_SIZE = 10
MIN_SAMPLES = 5

# One colour per cluster number, the same in every panel. Noise is grey.
PALETTE = [to_hex(plt.get_cmap("tab10")(i)) for i in range(10)]
NOISE_COLOUR = "#c8c8c8"
MIXED_COLOUR = "#808080"  # dendrogram links above the cut, which join several segments


def colour(label):
    return NOISE_COLOUR if label < 0 else PALETTE[label % len(PALETTE)]


def load_features(path):
    """The FEATURES columns of the CSV as an (n_customers, 2) float array."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return np.array([[float(row[name]) for name in FEATURES] for row in rows])


def match_labels(reference, labels):
    """Renumber `labels` so that each cluster takes the number of the `reference`
    cluster it overlaps most (a one-to-one matching). Noise (-1) stays -1. A cluster
    with no partner gets a new number after the reference numbers."""
    ref_ids = [r for r in np.unique(reference) if r >= 0]
    ids = [c for c in np.unique(labels) if c >= 0]
    overlap = np.array([[np.sum((labels == c) & (reference == r)) for r in ref_ids] for c in ids])
    rows, cols = linear_sum_assignment(-overlap)  # maximise the total overlap
    mapping = {ids[i]: ref_ids[j] for i, j in zip(rows, cols)}
    next_id = max(ref_ids, default=-1) + 1
    for c in ids:
        if c not in mapping:
            mapping[c], next_id = next_id, next_id + 1
    return np.array([mapping.get(c, -1) for c in labels])


def sklearn_tree(model):
    """scipy linkage matrix from a fitted scikit-learn AgglomerativeClustering
    (needs compute_distances=True): children_ and distances_, plus the cluster sizes."""
    n = len(model.labels_)
    sizes = np.ones(2 * n - 1)
    for i, (a, b) in enumerate(model.children_):
        sizes[n + i] = sizes[a] + sizes[b]
    return np.column_stack([model.children_, model.distances_, sizes[n:]]).astype(float)


def plot_dendrogram(ax, Z, labels, title):
    """Dendrogram with each segment in its scatter-plot colour. A link has the colour
    of a segment if all the customers below it are in that segment."""
    n = len(labels)
    node_label = list(labels) + [None] * (n - 1)  # segment of each node, None if mixed
    for i, (a, b) in enumerate(Z[:, :2].astype(int)):
        node_label[n + i] = node_label[a] if node_label[a] == node_label[b] else None
    dendrogram(Z, ax=ax, no_labels=True,
               link_color_func=lambda node: MIXED_COLOUR if node_label[node] is None
               else colour(node_label[node]))
    cut_height = (Z[-N_SEGMENTS, 2] + Z[-N_SEGMENTS + 1, 2]) / 2  # between the top merges
    ax.axhline(cut_height, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Customers")
    ax.set_ylabel("Ward distance")


def plot_clusters(ax, X, labels, title):
    """Scatter plot with one colour per cluster; noise (-1) in grey."""
    for label in np.unique(labels):
        members = labels == label
        name = "noise" if label < 0 else f"cluster {label + 1}"
        ax.scatter(X[members, 0], X[members, 1], s=25 if label < 0 else 40,
                   c=colour(label), label=f"{name} ({members.sum()})",
                   edgecolors="white", linewidths=0.4)
    ax.set_title(title)
    ax.set_xlabel(FEATURES[0])
    ax.set_ylabel(FEATURES[1])
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=pathlib.Path, default=DATA,
                        help=f"path of Mall_Customers.csv (default: {DATA})")
    parser.add_argument("--save", metavar="FILE", help="write the figure to FILE instead of showing it")
    args = parser.parse_args()

    X = load_features(args.data)
    print(f"{len(X)} customers, features: {', '.join(FEATURES)}")

    # cuhc: the Ward tree (scipy linkage format), the 5 Ward segments, and HDBSCAN.
    cuhc_tree = cuhc.linkage(X, method="ward")
    cuhc_ward = cuhc.AgglomerativeClustering(n_clusters=N_SEGMENTS, linkage="ward").fit(X).labels_
    cuhc_hdbscan = cuhc.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES).fit(X).labels_

    # scikit-learn: the same three results. compute_distances=True keeps the merge
    # heights, so the tree can be drawn.
    sk_model = SkAgglomerative(n_clusters=N_SEGMENTS, linkage="ward", compute_distances=True).fit(X)
    sk_tree = sklearn_tree(sk_model)
    sk_ward = match_labels(cuhc_ward, sk_model.labels_)
    sk_hdbscan_model = SkHDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES, copy=True)
    sk_hdbscan = match_labels(cuhc_hdbscan, sk_hdbscan_model.fit(X).labels_)

    # The values are integers, so many distances are equal. With ties, two correct
    # implementations can merge in a different sequence, so the agreement (adjusted
    # Rand index) can be a little below 1.
    for name, ours, theirs in [("Ward", cuhc_ward, sk_ward), ("HDBSCAN", cuhc_hdbscan, sk_hdbscan)]:
        n_clusters = len(set(ours) - {-1})
        print(f"{name:8} cuhc: {n_clusters} clusters, {np.sum(ours < 0)} noise | "
              f"scikit-learn: {len(set(theirs) - {-1})} clusters, {np.sum(theirs < 0)} noise | "
              f"agreement (adjusted Rand index) {adjusted_rand_score(theirs, ours):.3f}, "
              f"{np.sum(ours != theirs)} customers labelled differently")

    hdbscan_title = f"min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES}"
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    for row, (library, tree, ward, hdbscan) in enumerate([
        ("cuhc", cuhc_tree, cuhc_ward, cuhc_hdbscan),
        ("scikit-learn", sk_tree, sk_ward, sk_hdbscan),
    ]):
        plot_dendrogram(axes[row, 0], tree, ward,
                        f"{library}: Ward dendrogram (cut into {N_SEGMENTS} segments)")
        plot_clusters(axes[row, 1], X, ward, f"{library}: Ward, {N_SEGMENTS} segments")
        plot_clusters(axes[row, 2], X, hdbscan, f"{library}: HDBSCAN ({hdbscan_title})")
    fig.suptitle("Mall customers: cuhc (top) and scikit-learn (bottom)", fontsize=14)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=110)
        print(f"Figure written to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
