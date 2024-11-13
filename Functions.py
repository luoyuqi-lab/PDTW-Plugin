import numpy as np

def dist(a, b):
    return (a - b) ** 2

def fill_envelope(series, w):
    U = [0] * len(series)
    L = [0] * len(series)
    for i in range(len(series)):
        start = max(i-w, 0)
        stop = min(i+w, len(series)-1)
        U[i] = max(series[start:stop+1])
        L[i] = min(series[start:stop+1])

    return U, L

def segment_cont(a, b):
    return np.sum(np.abs(a - b))

def TS_segmentation(a, b, N, w):
    l = len(a) // N
    x1 = [a[max(0, i * l - (w + 1)):min((i + 1) * l, len(a))] for i in range(N)]
    y1 = [b[max(0, i * l - (w + 1)):min((i + 1) * l, len(b))] for i in range(N)]

    seg_cont_dis = [segment_cont(a[i * l:(i + 1) * l], b[i * l:(i + 1) * l]) for i in range(N)]
    sorted_indices = sorted(range(N), key=lambda k: seg_cont_dis[k], reverse=True)

    return x1, y1, sorted_indices

def txt_loader(filepath):
    x = []
    with open(str(filepath), 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split()]
            x.append(value)
    return x

def part_contri(a, b):
    """
    :param a: time series 1
    :param b: time series 2
    :return: sum of Euclidean distance difference of every data point.
    """
    c = [abs(a[i]-b[i]) for i in range(len(a))]
    Eucli_dis = sum(c)
    return Eucli_dis

def cdtw(a, b, r, return_path=True):
    """ Compute the DTW distance between 2 time series with a global window constraint (max warping degree)
    :param a: the time series array 1, template, in x-axis direction
    :param b: the time series array 2, sample, in y-axis direction
    :param r: the size of Sakoe-Chiba warping band
    :return: the DTW distance cost_prev[k]
             path: the optimal dtw mapping path
             M: Warping matrix
             D: Distance matrix (by squared Euclidean distance)
    """
    M = []
    m = len(a)
    k = 0
    cost = [float('inf')] * (2 * r + 1)
    cost_prev = [float('inf')] * (2 * r + 1)
    for i in range(0, m):
        k = max(0, r - i)
        for j in range(max(0, i - r), min(m - 1, i + r) + 1):
            # Initialize the first cell
            if i == 0 and j == 0:
                cost[k] = (a[0] - b[0]) ** 2
                k += 1
                continue
            y = float('inf') if j - 1 < 0 or k - 1 < 0 else cost[k - 1]
            x = float('inf') if i < 1 or k > 2 * r - 1 else cost_prev[k + 1]
            z = float('inf') if i < 1 or j < 1 else cost_prev[k]
            cost[k] = min(x, y, z) + (a[i] - b[j]) ** 2
            k += 1
        # Move current array (cost matrix) to previous array
        cost_prev = cost
        if return_path:
            M.append(cost_prev.copy())
    # The DTW distance is in the last cell in the cost matrix of size O(m^2) or !!At the middle of our array!!
    if return_path:
        i = m - 1
        j = r
        rj = m - 1
        path = [[m - 1, m - 1]]
        k -= 1
        while i != 0 or rj != 0:
            # From [n,m] to [0,0] in cost matrix to find optimal path
            x = M[i][j - 1] if j - 1 >= 0 else float('inf')
            y = M[i - 1][j] if i - 1 >= 0 else float('inf')
            z = M[i - 1][j + 1] if i - 1 >= 0 and j + 1 <= 2 * r else float('inf')
            # Save the real location index of optimal warping path point a_i mapping with point b_j.
            if min(x, y, z) == y:
                path.append([i - 1, rj - 1])
                i = i - 1
                rj = rj - 1
            elif min(x, y, z) == x:
                path.append([i, rj - 1])
                j = j - 1
                rj = rj - 1
            else:
                path.append([i - 1, rj])
                i = i - 1
                j = j + 1
        return cost_prev[k], path
    else:
        return cost_prev[k - 1]

def pdtw(a, b, r):
    """
    :param a: full template series
    :param b: full sample series
    :param r: max warping window / extra window size
    :return:
    """
    c = a[1:]
    d = b[1:]
    dis, p = cdtw(a[1:], b[1:], r, return_path=True)
    E = []
    path = []
    for i in range(len(p)):
        Euclidean_d = (c[p[i][0]] - d[p[i][1]])
        if a[0] == 0 and p[i][0] > len(c) - r - 2:
            E.append(p[i])
            dis -= Euclidean_d * Euclidean_d
        elif a[0] == 1 and (p[i][0] < r + 1 or p[i][0] > len(c) - r - 2):
            E.append(p[i])
            dis -= Euclidean_d * Euclidean_d
        elif a[0] == 2 and p[i][0] < r + 1:
            E.append(p[i])
            dis -= Euclidean_d * Euclidean_d
        else:
            path.append(p[i])
    return dis, path

def LB_Petitjean(q, t, w, ut, lt,  bsf):
    lb = 0
    istart = 0

    if w >= 1 and len(q) >= 6:
        q0, qe0, q1, q2 = q[0], q[-1], q[1], q[2]
        t0, te0, t1, t2 = t[0], t[-1], t[1], t[2]

        d01 = dist(q0, t1)
        d11 = dist(q1, t1)
        d10 = dist(q1, t0)

        if w == 1:
            lb = dist(q0, t0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01, d11) + dist(q1, t2),
                    min(d10, d11) + dist(q2, t1)
                )
            )
        else:
            lb = dist(q0, t0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01 + dist(q0, t2), min(d01, d11) + dist(q1, t2)),
                    min(d10 + dist(q2, t0), min(d10, d11) + dist(q2, t1))
                )
            )

        if lb > bsf:
            return lb

        q1, q2 = q[-2], q[-3]
        t1, t2 = t[-2], t[-3]

        d01 = dist(qe0, t1)
        d11 = dist(q1, t1)
        d10 = dist(q1, te0)

        if w == 1:
            lb += dist(qe0, te0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01, d11) + dist(q1, t2),
                    min(d10, d11) + dist(q2, t1)
                )
            )
        else:
            lb += dist(qe0, te0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01 + dist(qe0, t2), min(d01, d11) + dist(q1, t2)),
                    min(d10 + dist(q2, te0), min(d10, d11) + dist(q2, t1))
                )
            )

        if lb > bsf:
            return lb

        istart = 3

    proj = np.zeros(len(q))
    for i in range(istart, len(q) - istart):
        if lb > bsf:
            return lb
        qi = q[i]
        if qi > ut[i]:
            lb += dist(qi, ut[i])
            proj[i] = ut[i]
        elif qi < lt[i]:
            lb += dist(qi, lt[i])
            proj[i] = lt[i]
        else:
            proj[i] = qi

    up, lp = fill_envelope(proj, w)

    if lb > bsf:
        return lb

    for i in range(istart):
        proj[i] = q[i]
        proj[-i - 1] = q[-i - 1]
    uq, lq= fill_envelope(q, w)
    for i in range(istart, len(t) - istart):
        if lb > bsf:
            return lb
        ti = t[i]
        if ti > up[i]:
            if up[i] > uq[i]:
                lb += dist(ti, uq[i]) - dist(up[i], uq[i])
            else:
                lb += dist(ti, up[i])
        elif ti < lp[i]:
            if lp[i] < lq[i]:
                lb += dist(ti, lq[i]) - dist(lp[i], lq[i])
            else:
                lb += dist(ti, lp[i])

    return lb

def LB_Keogh(t, s, w, ub, lb, bsf):
    lb_dis = 0
    for i in range(len(t)):
        if lb_dis > bsf:
            return lb_dis
        ti = t[i]
        if ti > ub[i]:
            lb_dis += (ti - ub[i]) ** 2
        elif ti < lb[i]:
            lb_dis += (ti - lb[i]) ** 2

    return lb_dis

def LB_New(a, b, w, ub, lb, bsf):
    lb_dis = 0
    for i in range(len(a)):

        if lb_dis > bsf:
            return lb_dis
        ai = a[i]
        if ai > ub[i]:
            lb_dis += (ai - ub[i]) ** 2
        elif ai < lb[i]:
            lb_dis += (ai - lb[i]) ** 2
        else:
            left_bound = max(0, i - w)
            right_bound = min(len(b), i + w + 1)
            min_distance = float('inf')  # define min_distance
            for m in range(left_bound, right_bound):
                distance = (ai - b[m]) ** 2
                if distance < min_distance:
                    min_distance = distance
            lb_dis += min_distance
    return lb_dis

def LB_Webb(q, t, window, ut, lt, ult, lut, bsf):
    lb = 0
    istart = 0

    if window >= 1 and len(q) >= 6:
        q0 = q[0]
        t0 = t[0]
        qe0 = q[-1]
        te0 = t[-1]
        q1 = q[1]
        t1 = t[1]
        t2 = t[2]
        q2 = q[2]

        d01 = dist(q0, t1)
        d11 = dist(q1, t1)
        d10 = dist(q1, t0)

        if window == 1:
            lb = dist(q0, t0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01, d11) + dist(q1, t2),
                    min(d10, d11) + dist(q2, t1)
                )
            )
        else:
            lb = dist(q0, t0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01 + dist(q0, t2), min(d01, d11) + dist(q1, t2)),
                    min(d10 + dist(q2, t0), min(d10, d11) + dist(q2, t1))
                )
            )

        if lb > bsf:
            return lb

        q1 = q[-2]
        t1 = t[-2]
        t2 = t[-3]
        q2 = q[-3]

        d01 = dist(qe0, t1)
        d11 = dist(q1, t1)
        d10 = dist(q1, te0)

        if window == 1:
            lb += dist(qe0, te0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01, d11) + dist(q1, t2),
                    min(d10, d11) + dist(q2, t1)
                )
            )
        else:
            lb += dist(qe0, te0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01 + dist(qe0, t2), min(d01, d11) + dist(q1, t2)),
                    min(d10 + dist(q2, te0), min(d10, d11) + dist(q2, t1))
                )
            )

        if lb > bsf:
            return lb

        istart = 3
    freeCountAbove = window
    freeCountBelow = window

    qEnd = len(q) - istart
    lb_keogh1 = 0
    for i in range(istart, qEnd):
        if lb_keogh1 + lb > bsf:
            return lb_keogh1 + lb
        qi = q[i]
        if qi > ut[i]:
            lb_keogh1 += dist(qi, ut[i])
        elif qi < lt[i]:
            lb_keogh1 += dist(qi, lt[i])
    lb += lb_keogh1
    uq, lq = fill_envelope(q, window)
    uuq,luq = fill_envelope(uq, window)
    ulq,llq = fill_envelope(lq, window)
    for i in range(istart, qEnd):
        if lb > bsf:
            break
        qi = q[i]
        if qi > ut[i]:
            if ut[i] >= ulq[i]:
                freeCountBelow += 1
            else:
                freeCountBelow = 0
            freeCountAbove = 0
        elif qi < lt[i]:
            if lt[i] <= luq[i]:
                freeCountAbove += 1
            else:
                freeCountAbove = 0
            freeCountBelow = 0
        else:
            freeCountAbove += 1
            freeCountBelow += 1

        if i >= window + istart:
            j = i - window

            tj = t[j]
            uqj = uq[j]
            if tj > uqj:
                if freeCountAbove > 2 * window:
                    lb += dist(tj, uqj)
                else:
                    ultj = ult[j]
                    if tj > ultj and ultj >= uqj:
                        lb += dist(tj, uqj) - dist(ultj, uqj)
            else:
                lqj = lq[j]
                if tj < lqj:
                    if freeCountBelow > 2 * window:
                        lb += dist(tj, lqj)
                    else:
                        lutj = lut[j]
                        if tj < lutj and lutj <= lqj:
                            lb += dist(tj, lqj) - dist(lutj, lqj)

    for j in range(qEnd - window, qEnd):
        if lb > bsf:
            break

        tj = t[j]
        uqj = uq[j]
        if tj > uqj:
            if j >= qEnd - freeCountAbove + window:
                lb += dist(tj, uqj)
            else:
                ultj = ult[j]
                if tj > ultj and ultj >= uqj:
                    lb += dist(tj, uqj) - dist(ultj, uqj)
        else:
            lqj = lq[j]
            if tj < lqj:
                if j >= qEnd - freeCountBelow + window:
                    lb += dist(tj, lqj)
                else:
                    lutj = lut[j]
                    if tj < lutj and lutj <= lqj:
                        lb += dist(tj, lqj) - dist(lutj, lqj)

    return lb