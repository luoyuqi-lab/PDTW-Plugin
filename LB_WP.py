import math
from Functions import dist,cdtw, txt_loader, fill_envelope, segment_cont, TS_segmentation
import timeit
import numpy as np


def LB_WP_EA(a, b, w, ub, lb, ulb, lub, N, seg_num, bsf):
    if w >= 1 and len(a) >= 6:
        t0, te0, t1, t2 = a[0], a[-1], a[1], a[2]
        s0, se0, s1, s2 = b[0], b[-1], b[1], b[2]

        d01 = dist(t0, s1)
        d11 = dist(t1, s1)
        d10 = dist(t1, s0)

        if w == 1:
            first_3_dis = dist(t0, s0) + min(
                d11 + dist(t2, s2),
                min(
                    min(d01, d11) + dist(t1, s2),
                    min(d10, d11) + dist(t2, s1)
                )
            )
        else:
            first_3_dis = dist(t0, s0) + min(
                d11 + dist(t2, s2),
                min(
                    min(d01 + dist(t0, s2), min(d01, d11) + dist(t1, s2)),
                    min(d10 + dist(t2, s0), min(d10, d11) + dist(t2, s1))
                )
            )
        lb_dis = first_3_dis
        if lb_dis > bsf:
            return lb_dis

        t1, t2 = a[-2], a[-3]
        s1, s2 = b[-2], b[-3]

        d01 = dist(te0, s1)
        d11 = dist(t1, s1)
        d10 = dist(t1, se0)

        if w == 1:
            last_3_dis = dist(te0, se0) + min(
                d11 + dist(t2, s2),
                min(
                    min(d01, d11) + dist(t1, s2),
                    min(d10, d11) + dist(t2, s1)
                )
            )
        else:
            last_3_dis = dist(te0, se0) + min(
                d11 + dist(t2, s2),
                min(
                    min(d01 + dist(te0, s2), min(d01, d11) + dist(t1, s2)),
                    min(d10 + dist(t2, se0), min(d10, d11) + dist(t2, s1))
                )
            )
        lb_dis += last_3_dis
        if lb_dis > bsf:
            return lb_dis
    c, d, seg_con = TS_segmentation(a, b, N, w)
    l = math.floor(len(a) / N)
    lb_PW = lb_dis
    lb_K_temp = 0
    for j in range(N):
        if seg_con[j] == 0:
            start = 3
            stop = l
        elif seg_con[j] == N - 1:
            start = (seg_con[j]) * l
            stop = len(a) - 3
        else:
            start = (seg_con[j]) * l
            stop = start + l
        if j < seg_num:
            for i in range(start, stop):
                if lb_PW + lb_K_temp > bsf:
                    return lb_PW + lb_K_temp
                ai = a[i]
                if ai > ub[i]:
                    lb_K_temp += (ai - ub[i]) ** 2

                elif ai < lb[i]:
                    lb_K_temp += (ai - lb[i]) ** 2
        else:
            for i in range(start, stop):
                if lb_PW + lb_K_temp > bsf:
                    return lb_PW + lb_K_temp
                ai = a[i]
                if ai > ub[i]:
                    lb_PW += (ai - ub[i]) ** 2
                elif ai < lb[i]:
                    lb_PW += (ai - lb[i]) ** 2

    for p in range(seg_num):
        e, f = c[seg_con[p]][1:], d[seg_con[p]][1:]
        m = len(e)
        if seg_con[p] == 0:
            lb_pdtw = - first_3_dis
        elif seg_con[p] == N - 1:
            lb_pdtw = - last_3_dis
        else:
            lb_pdtw = 0
        cost = [float('inf')] * (2 * w + 1)
        cost_prev = [float('inf')] * (2 * w + 1)
        additional_window_dis = float('inf')
        for i in range(0, m):
            k = max(0, w - i)
            for j in range(max(0, i - w), min(m - 1, i + w) + 1):
                if i == 0 and j == 0:
                    cost[k] = (e[0] - f[0]) ** 2
                    k += 1
                    continue
                y = float('inf') if j - 1 < 0 or k - 1 < 0 else cost[k - 1]
                x = float('inf') if i < 1 or k > 2 * w - 1 else cost_prev[k + 1]
                z = float('inf') if i < 1 or j < 1 else cost_prev[k]
                cost[k] = min(x, y, z) + (e[i] - f[j]) ** 2
                k += 1
            if i == w + 1:
                index_min_row = cost.index(min(cost))
                additional_window_dis = min(cost_prev[index_min_row - 1], cost_prev[index_min_row],
                                            cost[index_min_row - 1])
            cost_prev = cost
            if i > w + 1:
                lb_pdtw = min(cost) - additional_window_dis
                if lb_PW + lb_pdtw > bsf:
                    return lb_PW + lb_pdtw
        lb_PW = lb_PW + lb_pdtw

    U1, L1 = fill_envelope(a, w)
    for i in range(3, len(a) - 3):
        if lb_dis > bsf:
            return lb_dis
        bi = b[i]
        if bi > U1[i]:
            lb_dis += (bi - U1[i]) ** 2
        elif bi < L1[i]:
            lb_dis += (bi - L1[i]) ** 2

    for j in range(seg_num):
        if seg_con[j] == 0:
            start = 3
            stop = l
        elif seg_con[j] == N - 1:
            start = (seg_con[j]) * l
            stop = len(a) - 3
        else:
            start = (seg_con[j]) * l
            stop = start + l
        for i in range(start, stop):
            bi = b[i]
            if bi > U1[i]:
                lb_dis -= (bi - U1[i]) ** 2
            elif bi < L1[i]:
                lb_dis -= (bi - L1[i]) ** 2
    if lb_dis + lb_pdtw > bsf:
        return lb_dis + lb_pdtw

    Uua, Lua = fill_envelope(U1, w)
    Ula, Lla = fill_envelope(L1, w)
    freeCountAbove = w
    freeCountBelow = w
    for f in range(seg_num, N):
        if seg_con[f] == 0:
            start = 3
            stop = l
        elif seg_con[f] == N - 1:
            start = (seg_con[f]) * l
            stop = len(a) - 3
        else:
            start = (seg_con[f]) * l
            stop = start + l
        for i in range(start, stop):
            if a[i] > ub[i]:
                if ub[i] >= Ula[i]:
                    freeCountBelow += 1
                else:
                    freeCountBelow = 0
                freeCountAbove = 0
            elif a[i] < lb[i]:
                if lb[i] <= Lua[i]:
                    freeCountAbove += 1
                else:
                    freeCountAbove = 0
                freeCountBelow = 0
            else:
                freeCountAbove += 1
                freeCountBelow += 1

            if i >= w + 3:
                j = i - 3

                tj = b[j]
                uqj = U1[j]
                if tj > uqj:
                    if freeCountAbove > 2 * w:
                        lb_PW += dist(tj, uqj)
                    else:
                        ultj = ulb[j]
                        if tj > ultj and ultj >= uqj:
                            lb_PW += dist(tj, uqj) - dist(ultj, uqj)
                else:
                    lqj = L1[j]
                    if tj < lqj:
                        if freeCountBelow > 2 * w:
                            lb_PW += dist(tj, lqj)
                        else:
                            lutj = lub[j]
                            if tj < lutj and lutj <= lqj:
                                lb_PW += dist(tj, lqj) - dist(lutj, lqj)

    for k in range(seg_num, N):
        if seg_con[k] == N - 1:
            for j in range(len(a) - w, len(a) - w):
                if lb_PW > bsf:
                    return lb_PW
                tj = b[j]
                uqj = U1[j]
                if tj > uqj:
                    if j >= 3 - freeCountAbove + w:
                        lb_PW += dist(tj, uqj)
                    else:
                        ultj = ulb[j]
                        if tj > ultj and ultj >= uqj:
                            lb_PW += dist(tj, uqj) - dist(ultj, uqj)
                else:
                    lqj = L1[j]
                    if tj < lqj:
                        if j >= 3 - freeCountBelow + w:
                            lb_PW += dist(tj, lqj)
                        else:
                            lutj = lub[j]
                            if tj < lutj and lutj <= lqj:
                                lb_PW += dist(tj, lqj) - dist(lutj, lqj)
        else:
            break
    return lb_PW

