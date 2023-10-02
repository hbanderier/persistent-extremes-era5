import math
import networkx as nx
from itertools import pairwise, permutations, combinations


def compute_distance_matrix_(points):
    points_weighed = points[:, [1, 0]]
    dist_matrix = haversine_distances(np.radians(points_weighed))
    if points.shape[1] <= 3:
        return dist_matrix
    return np.sqrt(dist_matrix ** 2 + (points[:, None, 2] - points[None, :, 2]) ** 2)


def find_jets_2D_(points, eps, kind="AgglomerativeClustering"):
    dist_matrix = compute_distance_matrix_(points)
    if kind == "DBSCAN":
        model = DBSCAN(
            eps=eps,
            metric="precomputed",
        )
    elif kind == "AgglomerativeClustering":  # strategy pattern ?
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=eps,
            metric="precomputed",
            linkage="single",
        )
    elif kind == "HDBSCAN":  # strategy pattern ?
        model = HDBSCAN(
            min_cluster_size=100,
            cluster_selection_epsilon=eps,
            metric='precomputed',
        )
    elif kind == "OPTICS":  # strategy pattern ?
        model = OPTICS(
            metric='precomputed',
        )
    labels = model.fit(dist_matrix).labels_
    masks = labels_to_mask(labels)
    return labels, masks


def find_jets(
    X,
    lon: NDArray,
    lat: NDArray,
    height=20,
    distance=20,
    width=6,
    grad=4,
    juncdistlo=5,
    juncdistla=6,
    cutoff=60,
) -> list:
    jets = []
    for i, x in enumerate(X.T):
        lo = lon[i]
        peaks = find_peaks(x, height=height, distance=distance, width=width)[0]
        for la, sp in zip(lat[peaks], x[peaks]):
            found = False
            for jet in jets:
                if (np.abs(jet[-1][0] - lo) + np.abs(jet[-1][1] - la)) <= grad:
                    jet.append((lo, la, sp))
                    found = True
                    break
            if not found:
                jets.append([(lo, la, sp)])

    jets = [np.asarray(jet) for jet in jets]
    jets = [jet[np.argsort(jet[:, 0])] for jet in jets]
    done = False
    while not done:
        done = True
        njets = len(jets)
        for i1, i2 in permutations(range(njets), 2):
            jet1, jet2 = jets[i1], jets[i2]
            dlo1 = np.abs(jet1[-1][0] - jet2[0][0])
            dla1 = np.abs(jet1[-1][1] - jet2[0][1])
            dlo2 = np.abs(jet1[-1][0] - jet2[-1][0])
            dla2 = np.abs(jet1[-1][1] - jet2[-1][1])
            if dlo1 <= juncdistlo and dla1 <= juncdistla:
                jets[i1] = np.concatenate([jet1, jet2], axis=0)
                jets[i1] = jets[i1][np.argsort(jets[i1][:, 0])]
                del jets[i2]
                done = False
                break
            if dlo2 <= juncdistlo and dla2 <= juncdistla:
                jets[i1] = np.concatenate([jet1, jet2[::-1]], axis=0)
                jets[i1] = jets[i1][np.argsort(jets[i1][:, 0])]
                del jets[i2]
                done = False
                break

    for i in range(len(jets) - 1, -1, -1):
        if len(jets[i]) < cutoff:
            del jets[i]

    return jets


def compute_distance_matrix_(points, weights=None):
    if weights is None:
        weights = [1, 1]
    points_weighed = points[:, [1, 0]] * np.asarray(weights)[None, :]
    return haversine_distances(np.radians(points_weighed))


def find_jets_v2_(
    points,
    eps,
    weights=None,
    kind = 'AgglomerativeClustering'
):
    dist_matrix = compute_distance_matrix_(points, weights=weights)
    if kind == 'DBSCAN':
        model = DBSCAN(
            eps=eps,
            metric="precomputed",
        )
    elif kind == 'AgglomerativeClustering': # strategy pattern ?
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=eps,
            metric="precomputed",
            linkage='single'
        )
    labels = model.fit(dist_matrix).labels_
    masks = labels_to_mask(labels)
    return [points[mask] for mask in masks.T]


def find_jets_v2(
    X,
    lon: NDArray,
    lat: NDArray,
    height=20,
    eps=3,
    cutoff=1900,
    weights=None,
    kind: str = 'DBSCAN',
    remove_outliers: int = 0,
) -> list:
    points = []
    res = lon[1] - lon[0]
    cutoff = cutoff / res
    for i, x in enumerate(X.T):
        lo = lon[i]
        peaks = find_peaks(x, height=height, distance=10, width=1)[0]
        for peak in peaks:
            for plus in [-2, -1, 0, 1, 2]:
                try:
                    if x[peak + plus] > height:
                        points.append([lo, lat[peak + plus], x[peak + plus]])
                except IndexError:
                    pass
    # for j, x in enumerate(X):
    #     if j % 2 == 1:
    #         continue
    #     la = lat[j]
    #     peaks = find_peaks(x, height=height, distance=20000, width=5)[0]
    #     for peak in peaks:
    #         for plus in [-2, -1, 0, 1, 2]:
    #             try:
    #                 if x[peak + plus] > height:
    #                     points.append([lon[peak + plus], la, x[peak + plus]])
    #             except IndexError:
    #                 pass
    if len(points) == 0:
        return []
    points = np.atleast_2d(points)
    argsort = np.argsort(points[:, 0])
    points = points[argsort, :]
    dist_matrix = compute_distance_matrix_(points, weights=[1, 1])
    eps  = eps * np.radians(res)
    if remove_outliers:
        
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=np.radians(res) * 2.1, metric='precomputed', linkage='single').fit(dist_matrix).labels_
        masks = labels_to_mask(labels)
        groups = [points[mask] for mask in masks.T]
        
        points = np.concatenate([group for group in groups if len(group) > remove_outliers / res])
        argsort = np.argsort(points[:, 0])
        points = points[argsort, :]
    
    potential_jets = find_jets_v2_(points, eps, weights=weights, kind=kind)    
    k = 0
    this_eps = eps
    for i, jet in enumerate(potential_jets):
        jet = potential_jets[i]
        strength = np.sum(jet[:, 2])
        strong_enough = strength > cutoff
        
        diffx = np.abs(np.diff(jet[:, 0]))
        diffy = np.abs(np.diff(jet[:, 1]))
        jumpsx = np.sum(diffx[diffx > 2 * res])
        jumpsy = np.sum(diffy[diffy > 6 * res])
        too_split = jumpsx > 4 * res or jumpsy > 6 * 10 * res
        if not strong_enough:
            potential_jets[i] = None
        if all((j is None for j in potential_jets)):
            break
        if strong_enough and too_split and this_eps >= min_eps:
            k += 1
            this_eps = eps / (k / 2 + 1)
            potential_jets[i] = None
            potential_jets.extend(find_jets_v2_(jet, this_eps, kind))
        print(k, f'{this_eps:.2f}', f'{jumpsx:.2f}', f'{jumpsy:.2f}', f'{strength:.2f}')    
    jets = [j for j in potential_jets if j is not None]
    return jets


def weight(X, i, j, k, l, maxX):
    s1 = X[i, j]
    s2 = X[k, l]
    x = 1 - (s1 + s2) / maxX / 2
    return 1 - np.exp(-x ** 2 / 0.5)


def is_nei(i, j, k, l):
    return abs(k - i) <= 1 and abs(l - j) <= 1


def find_jets_v3(X, lon, lat, len_cutoff=20):
    sps = []
    maxX = np.amax(X)
    mask = (X > 20).ravel(order='F')
    grid = np.stack(np.meshgrid(np.arange(X.shape[0]), np.arange(X.shape[1])), axis=-1).reshape(-1, 2)
    masked_grid = grid[mask]
    labels = DBSCAN(eps=math.sqrt(2) * 1.1, metric='euclidean').fit(masked_grid).labels_
    masks = labels_to_mask(labels)
    groups = [masked_grid[mask] for mask in masks.T if np.sum(mask) > len_cutoff]
    for group in groups:
        G = nx.empty_graph(0)

        G.add_nodes_from((i, j) for (i, j) in group)
        G.add_weighted_edges_from(((i, j), (k, l), weight(X, i, j, k, l, maxX)) for ((i, j), (k, l)) in combinations(group, 2) if is_nei(i, j, k, l))

        minlon, maxlon = np.amin(group[:, 1]), np.amax(group[:, 1])
        lats1 = group[group[:, 1] == minlon, 0]
        edges1 = [0, *np.where(np.diff(lats1) > 6)[0] + 1, len(lats1) - 1]
        origins = []
        for i1, i2 in pairwise(edges1):
            origins.append((lats1[i1 + np.argmax(X[lats1[i1:i2 + 1], minlon])], minlon))
            
        lats2 = group[group[:, 1] == maxlon, 0]
        edges2 = [0, *np.where(np.diff(lats2) > 6)[0] + 1, len(lats2) - 1]
        dests = []
        for i1, i2 in pairwise(edges2):
            dests.append((lats2[i1 + np.argmax(X[lats2[i1:i2 + 1], maxlon])], maxlon))
            
        for orig, dest in product(origins, dests):
            sp = np.asarray(nx.shortest_path(G, orig, dest, weight='weight'))
            sp = np.asarray([lon[sp[:, 1]], lat[sp[:, 0]], X[sp[:, 0], sp[:, 1]]]).T
            sps.append(sp)
    return sps


def update_jet(jet: NDArray):
    x, y, s = jet.T
    x_, c_ = np.unique(x, return_index=True)
    s_split = np.split(s, c_[1:])
    y_ = np.empty(len(s_split))
    s_ = np.empty(len(s_split))
    for i, sp in enumerate(s_split):
        j = np.argmax(sp) + c_[i]
        y_[i] = y[j]
        s_[i] = s[j]
    return np.stack([x_, y_, s_], axis=-1)


def find_jets_2D(
    da: xr.DataArray, height=15, eps: float = 10, sigmas: Iterable = None, cutoff: float = 1500, kind: str = 'DBSCAN'
) -> list:
    if sigmas is None:
        sigmas = range(8, 24, 5)
    lon, lat = da.lon.values, da.lat.values
    res = lon[1] - lon[0]
    cutoff = cutoff / res
    X = da.values
    Xmax = X.max()
    X_norm = X / Xmax
    X_prime = frangi(X_norm, black_ridges=False, sigmas=sigmas, cval=1) * Xmax
    points = []
    for i, x in enumerate(X_prime.T):
        lo = lon[i]
        peaks = find_peaks(x, height=height, distance=10, width=2)[0]
        for peak in peaks:
            points.append([lo, lat[peak], X[peak, i]])
    for j, x in enumerate(X_prime):
        la = lat[j]
        peaks = find_peaks(x, height=height, distance=10, width=2)[0]
        for peak in peaks:
            points.append([lon[peak], la, X[j, peak]])
    if len(points) == 0:
        return []
    points = np.atleast_2d(points)
    argsort = np.argsort(points[:, 0])
    points = points[argsort, :]
    eps = eps * np.radians(res)
    labels, masks = find_jets_2D_(points, eps=eps, kind=kind) 
    potential_jets = [points[mask] for mask in masks.T]

    jets = [jet for jet in potential_jets if np.sum(jet[:, 2]) > cutoff]
    sorted_order = np.argsort(
        [np.average(jet[:, 1], weights=jet[:, 2]) for jet in jets]
    )
    return [update_jet(jets[i]) for i in sorted_order]


def find_jets_3D(da: xr.DataArray, height=20, eps: float = 10, sigmas: Iterable = None, vert_factor=2, kind: str = 'HDBSCAN'):
    if sigmas is None:
        sigmas = range(10, 21, 5)
    alts = vert_factor * metpy.calc.pressure_to_height_std(da.lev).values
    lon, lat = da.lon.values, da.lat.values
    all_points = []
    for il, X in enumerate(da.values):
        points = []
        Xmax = X.max()
        X_norm = X / Xmax
        X_prime = frangi(X_norm, black_ridges=False, sigmas=range(10, 21, 5), cval=1) * Xmax
        for i, x in enumerate(X_prime.T):
            lo = lon[i]
            peaks = find_peaks(x, height=height, distance=10, width=1)[0]
            for peak in peaks:
                points.append([lo, lat[peak], alts[il], X[peak, i]])
        for j, x in enumerate(X_prime):
            la = lat[j]
            peaks = find_peaks(x, height=height, distance=20000, width=5)[0]
            for peak in peaks:
                points.append([lon[peak], la, alts[il], X[j, peak]])
        points = np.atleast_2d(points)
        argsort = np.argsort(points[:, 0])
        points = points[argsort, :]
        all_points.append(points)
    all_points = np.concatenate(all_points)
    
    labels, masks = find_jets_2D_(all_points, eps, kind=kind)
        
    split_idx = 1 + np.where(np.diff(all_points[:, 2]))[0]
    split_labels = np.split(labels, split_idx)
    max_label = np.amax(labels)
    for il, labels in enumerate(split_labels):
        unique, idxs, counts = np.unique(
            labels, return_counts=True, return_index=True
        )
        for un, co, idx in zip(unique, counts, idxs):
            if (co > 30) or (un == -1):
                continue
            split_labels[il][idx] = -1
            max_label = max_label + 1
            if il == 0:
                continue
            for iln in range(il + 1, len(split_labels)):
                split_labels[iln][split_labels[iln] == un] = max_label
                
    split_points = np.split(all_points, split_idx)
    split_masks = [labels_to_mask(labels) for labels in split_labels]  
    jets = [[points[mask] for mask in masks.T] for masks, points in zip(split_masks, split_points)]
    return jets, split_labels 