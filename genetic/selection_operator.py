import numpy as np


def rouletteselection(_a, k):
    a = np.asarray(_a)
    idx = np.argsort(a)
    idx = idx[::-1]
    sort_a = a[idx]
    sum_a = np.sum(a).astype(np.float32)
    selected_indices = []
    for i in range(k):
        u = np.random.rand() * sum_a
        sum_ = 0
        for l in range(sort_a.shape[0]):
            sum_ += sort_a[l]
            if sum_ > u:
                selected_indices.append(idx[l])
                break
    return selected_indices


def numpy_rouletteselection(_a, k):
    return np.random.choice(_a, size=k, replace=True).tolist()


if __name__ == '__main__':
    a = [1, 3, 2, 1, 4, 4, 5]
    selected_index = rouletteselection(a, k=10000)

    new_a = [a[i] for i in selected_index]
    print(np.unique(new_a, return_counts=True))
    print(np.unique(numpy_rouletteselection(a, k=10000), return_counts=True))
