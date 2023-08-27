import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from numba import njit, jit

@jit
def track_jets(all_jets, all_props):
    factor = 0.2
    yearbreaks: int = 92
    i0 = 0
    il = len(all_jets)
    all_jets_over_time = [None] * len(all_jets[0])
    for j, (jet, prop) in enumerate(zip(all_jets[i0], all_props[i0])):
        all_jets_over_time[j] = [j, {i0: (jet, prop)}]
    flags = [list(range(len(all_jets_over_time)))]
    last_flag = flags[0][-1]
    for tp, jets in enumerate(all_jets[i0 + 1:il]): # can't really parallelize
        t = tp + i0
        flags.append([0] * len(jets))
        potentials = np.empty(len(all_jets_over_time), dtype=bool)
        for j, jtt in enumerate(all_jets_over_time):
            cond1 = t in jtt[1]
            cond2 = (t - 1 in jtt[1]) and ((t + 1) // yearbreaks == (t - 1) // yearbreaks)
            cond3 = (t - 1 in jtt[1]) and ((t + 1) // yearbreaks == (t - 1) // yearbreaks)
            potentials[j] = cond1 or cond2 or cond3
        jets_to_try = np.nonzero(potentials)[0]
        dist_mat = np.zeros((len(jets_to_try), len(jets)))
        if (t + 1) % yearbreaks == 0:
            dist_mat = np.full((len(jets_to_try), len(jets)), fill_value=10 * factor)
        else:
            for i, jtt_idx in enumerate(jets_to_try):
                jet_to_try = all_jets_over_time[jtt_idx]
                
                jet_to_try_ = list(jet_to_try[1].values())[-1][0][:, :2]
                for j, jet in enumerate(jets):
                    distances = haversine_distances(np.radians(jet_to_try_), np.radians(jet[:, :2])) 
                    dist_mat[i, j] = np.mean([
                        np.sum(np.amin(distances / len(jet_to_try_), axis=1)), 
                        np.sum(np.amin(distances / len(jet), axis=0))
                    ])
        connected_mask = dist_mat < factor
        flagged = [1000] # can't have empty lists
        for i, jtt_idx in enumerate(jets_to_try):
            jet_to_try = all_jets_over_time[jtt_idx]
            js = np.argsort(dist_mat[i])
            for j in js:
                if not connected_mask[i, j]:
                    break
                if j in flagged:
                    continue
                jet_to_try[1][t + 1] = (jets[j], all_props[t + 1][j])
                flagged.append(j)
                flags[-1][j] = jet_to_try[0]
                break
                
            
        for j, jet in enumerate(jets):
            if j not in flagged:
                last_flag += 1
                all_jets_over_time.append([last_flag, {t + 1: (jet, all_props[t + 1][j])}]) 
                flags[-1][j] = last_flag
    return all_jets_over_time, flags


@njit
def isin(a, b):
    shape = a.shape
    a = a.ravel()
    n = len(a)
    result = np.full(n, False, dtype=np.bool_)
    set_b = set(b)
    for i in range(n):
        if a[i] in set_b:
            result[i] = True
    return result.reshape(shape)

@njit
def amin_ax0(a):
    result = np.zeros(a.shape[1])
    for i, a_ in enumerate(a.T):
        result[i] = np.amin(a_)
    return result

@njit
def amin_ax1(a):
    result = np.zeros(a.shape[0])
    for i, a_ in enumerate(a):
        result[i] = np.amin(a_)
    return result

@njit
def track_jets_numba(all_jets_one_array, where_are_jets, maxnjets, num_indiv_jets):
    factor: float = 0.2
    yearbreaks: int = 92
    all_jets_over_time = np.full((12000, yearbreaks, 2), fill_value=len(where_are_jets), dtype=np.int32)
    last_valid_idx = np.full(12000, fill_value=yearbreaks, dtype=np.int32)
    for j in range(np.sum(where_are_jets[0, 0] >= 0)):
        all_jets_over_time[j, 0, :] = (0, j)
        last_valid_idx[j] = 0
    flags = np.full((len(where_are_jets), maxnjets), fill_value=num_indiv_jets, dtype=np.int32)
    last_flag = np.sum(where_are_jets[0, 0] >= 0) - 1
    for t, jet_idxs in enumerate(where_are_jets[1:]): # can't really parallelize
        potentials = np.zeros(50, dtype=np.int32)
        from_ = max(0, last_flag - 30)
        times_to_test = np.take_along_axis(all_jets_over_time[from_:last_flag + 1, :, 0], last_valid_idx[from_:last_flag + 1, None], axis=1).flatten()
        potentials = from_ + np.where(
            isin(times_to_test, [t, t - 1, t - 2]) &
            ((times_to_test // yearbreaks) == ((t + 1) // yearbreaks)).astype(np.bool_)
        )[0]
        num_valid_jets = np.sum(jet_idxs[:, 0] >= 0)
        dist_mat = np.zeros((len(potentials), num_valid_jets), dtype=np.float32)
        for i, jtt_idx in enumerate(potentials):
            t_jtt, j_jtt = all_jets_over_time[jtt_idx, last_valid_idx[jtt_idx]]
            k_jtt, l_jtt = where_are_jets[t_jtt, j_jtt]
            jet_to_try = all_jets_one_array[k_jtt:l_jtt, :2]
            for j in range(num_valid_jets):
                k, l = jet_idxs[j]
                jet = all_jets_one_array[k:l, :2]
                distances = np.sqrt(np.sum((np.radians(jet_to_try)[None, :, :] - np.radians(jet)[:, None, :]) ** 2, axis=-1))
                # distances = haversine_distances(np.radians(jet_to_try), np.radians(jet)) 
                dist_mat[i, j] = np.mean(np.array([
                    np.sum(amin_ax1(distances / len(jet_to_try))), 
                    np.sum(amin_ax0(distances / len(jet)))
                ]))
        connected_mask = dist_mat < factor
        flagged = np.zeros(num_valid_jets, dtype=np.bool_)
        for i, jtt_idx in enumerate(potentials):
            k_jtt, l_jtt = all_jets_over_time[jtt_idx, last_valid_idx[jtt_idx]]
            jet_to_try = all_jets_one_array[k_jtt:l_jtt, :2]
            js = np.argsort(dist_mat[i])
            for j in js:
                if not connected_mask[i, j]:
                    break
                if flagged[j]:
                    continue
                last_valid_idx[jtt_idx] = last_valid_idx[jtt_idx] + 1
                all_jets_over_time[jtt_idx, last_valid_idx[jtt_idx], :] = (t + 1, j)
                flagged[j] = True
                flags[t + j, j] = jtt_idx
                break
                
        for j in range(num_valid_jets):
            if not flagged[j]:
                last_flag += 1
                all_jets_over_time[last_flag, 0, :] = (t + 1, j)
                last_valid_idx[last_flag] = 0
                flags[t + j, j] = last_flag
                flagged[j] = True

    return all_jets_over_time, flags


def track_jets_numpy(all_jets_one_array, where_are_jets, maxnjets, num_indiv_jets): # no numba, numpy version
    factor: float = 0.2
    yearbreaks: int = 92
    n_jet_guess = 2000
    all_jets_over_time = np.full((n_jet_guess, yearbreaks, 2), fill_value=len(where_are_jets), dtype=np.int32)
    last_valid_idx = np.full(n_jet_guess, fill_value=yearbreaks, dtype=np.int32)
    for j in range(np.sum(where_are_jets[0, 0] >= 0)):
        all_jets_over_time[j, 0, :] = (0, j)
        last_valid_idx[j] = 0
    flags = np.full((len(where_are_jets), maxnjets), fill_value=num_indiv_jets, dtype=np.int32)
    last_flag = np.sum(where_are_jets[0, 0] >= 0) - 1
    for t, jet_idxs in enumerate(where_are_jets[1:]): # can't really parallelize
        potentials = np.zeros(50, dtype=np.int32)
        from_ = max(0, last_flag - 30)
        times_to_test = np.take_along_axis(all_jets_over_time[from_:last_flag + 1, :, 0], last_valid_idx[from_:last_flag + 1, None], axis=1).flatten()
        potentials = from_ + np.where(
            np.isin(times_to_test, [t, t - 1, t - 2]) &
            ((times_to_test // yearbreaks) == ((t + 1) // yearbreaks)).astype(np.bool_)
        )[0]
        num_valid_jets = np.sum(jet_idxs[:, 0] >= 0)
        dist_mat = np.zeros((len(potentials), num_valid_jets), dtype=np.float32)
        for i, jtt_idx in enumerate(potentials):
            t_jtt, j_jtt = all_jets_over_time[jtt_idx, last_valid_idx[jtt_idx]]
            k_jtt, l_jtt = where_are_jets[t_jtt, j_jtt]
            jet_to_try = all_jets_one_array[k_jtt:l_jtt, :2]
            for j in range(num_valid_jets):
                k, l = jet_idxs[j]
                jet = all_jets_one_array[k:l, :2]
                distances = haversine_distances(np.radians(jet_to_try), np.radians(jet)) 
                dist_mat[i, j] = np.mean(np.array([
                    np.sum(np.amin(distances / len(jet_to_try), axis=1)), 
                    np.sum(np.amin(distances / len(jet)), axis=0)
                ]))
        connected_mask = dist_mat < factor
        flagged = np.zeros(num_valid_jets, dtype=np.bool_)
        for i, jtt_idx in enumerate(potentials):
            k_jtt, l_jtt = all_jets_over_time[jtt_idx, last_valid_idx[jtt_idx]]
            jet_to_try = all_jets_one_array[k_jtt:l_jtt, :2]
            js = np.argsort(dist_mat[i])
            for j in js:
                if not connected_mask[i, j]:
                    break
                if flagged[j]:
                    continue
                last_valid_idx[jtt_idx] = last_valid_idx[jtt_idx] + 1
                all_jets_over_time[jtt_idx, last_valid_idx[jtt_idx], :] = (t + 1, j)
                flagged[j] = True
                flags[t + j, j] = jtt_idx
                break
                
        for j in range(num_valid_jets):
            if not flagged[j]:
                last_flag += 1
                all_jets_over_time[last_flag, 0, :] = (t + 1, j)
                last_valid_idx[last_flag] = 0
                flags[t + j, j] = last_flag
                flagged[j] = True

    return all_jets_over_time, flags


def extract_props_over_time_old(jet):
    timesteps = np.asarray(list(jet.keys()))
    varnames = list(jet[timesteps[0]][1].keys())
    props_over_time = {varname: np.zeros(len(timesteps)) for varname in varnames}
    for varname in varnames:
        for t, timestep in enumerate(timesteps):
            props_over_time[varname][t] = jet[timestep][1][varname]
    return props_over_time