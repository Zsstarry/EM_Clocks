import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from time import time
from numba import jit


@jit(nopython=True)
def DVTH(Fai, Theta):
    dvth = 0
    for i in range(n):
        for j in range(n):
            hs = np.sqrt(s[i, j] * s[i, j] + 2 *
                         lp2 * (1 - np.cos(Fai[i] - Fai[j])) + 8 * l * r * np.sin((Fai[i] - Fai[j]) / 2) * np.sin(
                (alpha[i] - alpha[j]) / 2) * np.sin((Fai[i] + Fai[j] - alpha[i] - alpha[j]) / 2 - Theta))
            # print(hs)
            if Aij[i, j] != 0:
                dvth += 2 * l * r * kfai * Aij[i, j] * (1 - s[i, j] / hs) * np.sin(
                    (Fai[i] - Fai[j]) / 2) * np.sin((alpha[j] - alpha[i]) / 2) * np.cos(
                    ((Fai[i] + Fai[j] - alpha[i] - alpha[j]) / 2 - Theta))
    return dvth


@jit(nopython=True)
def DVFAI(Fai, Theta, ith):
    dvfai = 0
    for j in range(n):
        hs = np.sqrt(s[ith, j] * s[ith, j] + 2 *
                     lp2 * (1 - np.cos(Fai[ith] - Fai[j])) + 8 * l * r * np.sin((Fai[ith] - Fai[j]) / 2) * np.sin(
            (alpha[ith] - alpha[j]) / 2) * np.sin((Fai[ith] + Fai[j] - alpha[ith] - alpha[j]) / 2 - Theta))
        # print(hs)
        if Aij[ith, j] != 0:
            dvfai += Aij[ith, j] * kfai * l * (1 - s[ith, j] / hs) * (
                l * np.sin(Fai[ith] - Fai[j]) + 2 * r * np.sin((alpha[ith] - alpha[j]) / 2) *
                np.sin(Fai[ith] - (alpha[ith] + alpha[j]) / 2 - Theta))
    return dvfai


def dXdt(t, X):
    # x[0] = 0  # th
    # x[1:(n + 1)] = fai  # fai
    # x[n + 1] = 0  # omega
    # x[(n + 2):] = w  # w
    A[:(n + 1), :(n + 1)] = np.eye(n + 1)
    A[n + 1, n + 1] = B0 + n * m * rp2
    A[n + 1, 0] = cth
    A[(n + 2):, 1:(n + 1)] = np.eye(n) * cfai
    A[(n + 1), (n + 2):] = np.array([m * r * l * np.sin(X[i + 1] - X[0] - alpha[i])
                                     for i in range(n)])
    A[(n + 2):, (n + 1)] = A[(n + 1), (n + 2):]
    A[(n + 2):, (n + 2):] = np.eye(n) * m * lp2

    B[0] = X[n + 1]
    B[1:(n + 1)] = X[(n + 2):]
    b_sum = 0
    b_rest = np.zeros(n)
    for i in range(n):
        b_sum += m * r * l * np.power(X[n + 2 + i], 2) * np.cos(X[i + 1] -
                                                                X[0] - alpha[i]) + m * r * g * np.cos(alpha[i] + X[0])
        b_rest[i] = -m * g * l * np.sin(X[i + 1]) + m * r * l * np.power(
            X[n + 1], 2) * np.cos(X[i + 1] - X[0] - alpha[i]) - DVFAI(X[1:(n + 1)], X[0], i) + ME[i]

    B[n + 1] = -kth * X[0] - b_sum - \
        DVTH(X[1:(n + 1)], X[0])  # DVTH needs all fai
    B[(n + 2):] = b_rest

    # make sure A is not singular
    return np.linalg.inv(A).dot(B)


def positive_zero(i, Flag):
    def event(t, X):
        fai = (X[i + 1] % (2 * np.pi)) - \
            ((X[i + 1] % (2 * np.pi)) // np.pi) * (2 * np.pi)
        return fai
    event.terminal = Flag
    event.direction = 1
    return event


def negative_zero(i, Flag):
    def event(t, X):
        fai = (X[i + 1] % (2 * np.pi)) - \
            ((X[i + 1] % (2 * np.pi)) // np.pi) * (2 * np.pi)
        return fai
    event.terminal = Flag
    event.direction = -1
    return event


def positive_epsilon(i, Flag):
    def event(t, X):
        fai = (X[i + 1] % (2 * np.pi)) - \
            ((X[i + 1] % (2 * np.pi)) // np.pi) * (2 * np.pi)
        return fai - epsilon
    event.terminal = Flag
    event.direction = 1
    return event


def negative_epsilon(i, Flag):
    def event(t, X):
        fai = (X[i + 1] % (2 * np.pi)) - \
            ((X[i + 1] % (2 * np.pi)) // np.pi) * (2 * np.pi)
        return fai + epsilon
    event.terminal = Flag
    event.direction = -1
    return event


def positive_poincare(i, Flag):
    def event(t, X):
        theta = (X[i] % (2 * np.pi)) - \
            ((X[i] % (2 * np.pi)) // np.pi) * (2 * np.pi)
        return theta
    event.terminal = Flag
    event.direction = 1
    return event


def solution(p_init, p_ptr):
    init_fai = p_init
    x = np.zeros(d)
    # events detection
    find_y = []
    for i in range(n):
        find_y.append(positive_epsilon(i, True))
    for i in range(n):
        find_y.append(negative_epsilon(i, True))
    for i in range(n):
        find_y.append(positive_zero(i, True))
    for i in range(n):
        find_y.append(negative_zero(i, True))
    find_y.append(positive_poincare(p_ptr, True))
    # for iterating
    iteration = 0
    mini = 0.01
    c_y = 0
    c_pocr = 0
    ini_t = 0
    end_t = TIME
    interval = STEPS
    te_ttl = np.linspace(ini_t, end_t, interval)
    y = np.zeros((d + 1, 10000000))  # don't use float32, otherwise
    pocr_y = np.zeros((d + 1, 10000000))  # to [-π, π] does not
    t1 = time()
    while True:
        iteration += 1
        # initialization =====================
        if iteration == 1:
            x[0] = 0.01  # th
            x[1: n + 1] = init_fai
            x[n + 1] = 0  # omega
            x[(n + 2):] = 0  # w
            for i in range(n):
                fai = (x[i + 1] % (2 * np.pi)) - \
                    ((x[i + 1] % (2 * np.pi)) // np.pi) * (2 * np.pi)
                while abs(fai) > np.pi:
                    fai = (fai % (2 * np.pi)) - \
                        ((fai % (2 * np.pi)) // np.pi) * (2 * np.pi)
                if fai >= epsilon:
                    sigma[i] = 2
                elif fai <= -epsilon:
                    sigma[i] = 1
                else:
                    sigma[i] = 0
                if sigma[i] == 1 and 0 < fai < epsilon:
                    ME[i] = M
                elif sigma[i] == 2 and -epsilon < fai < 0:
                    ME[i] = -M
                else:
                    ME[i] = 0
        else:
            for i in range(n):
                fai = (x[i + 1] % (2 * np.pi)) - \
                    ((x[i + 1] % (2 * np.pi)) // np.pi) * (2 * np.pi)
                while abs(fai) > np.pi:
                    fai = (fai % (2 * np.pi)) - \
                        ((fai % (2 * np.pi)) // np.pi) * (2 * np.pi)
                if fai > epsilon and x[n + 2 + i] > 0:
                    sigma[i] = 2
                    ME[i] = 0
                elif fai < -epsilon and x[n + 2 + i] < 0:
                    sigma[i] = 1
                    ME[i] = 0
                elif fai > 0 and x[n + 2 + i] > 0:
                    if sigma[i] == 1:
                        ME[i] = M
                elif fai < 0 and x[n + 2 + i] < 0:
                    if sigma[i] == 2:
                        ME[i] = -M

        # modeling ===========================
        ts_solm = [ini_t, end_t]
        te_solm = te_ttl[(te_ttl - ini_t) >= 0]
        solm = solve_ivp(dXdt,
                         t_span=ts_solm,
                         y0=x,
                         t_eval=te_solm,
                         events=find_y)
        lt = solm.t.shape[0]
        y[-1, c_y:c_y + lt] = solm.t
        print(solm.t[-1])
        y[:d, c_y:c_y + lt] = solm.y
        c_y += lt

        if solm.status == 1:
            # the current position ================
            et = solm.t_events
            ey = solm.y_events
            for ei, v in enumerate(et[:-1]):
                if v.shape[0] != 0:
                    ini_t = v[0]
                    x = ey[ei][0]
                    break

            pocr_fai = (x[p_ptr] % (2 * np.pi)) - \
                ((x[p_ptr] % (2 * np.pi)) // np.pi) * (2 * np.pi)
            if abs(pocr_fai) < 0.000001 and x[n + p_ptr + 1] > 0:
                pocr_y[-1, c_pocr:c_pocr + 1] = ini_t
                pocr_y[:d, c_pocr:c_pocr + 1] = x.reshape(d, 1)
                c_pocr += 1
            elif et[-1].shape[0] != 0 and et[-1][0] != 0:
                ini_t = et[-1][0]
                x = ey[-1][0]
                pocr_y[-1, c_pocr:c_pocr + 1] = ini_t
                pocr_y[:d, c_pocr:c_pocr + 1] = x.reshape(d, 1)
                c_pocr += 1

            # forward a few steps ================
            t_fwd = te_ttl[(te_ttl - ini_t) > 0][0]
            ts_fwd = [ini_t, t_fwd]
            te_fwd = np.linspace(ini_t, t_fwd, 2)
            sol_fwd = solve_ivp(dXdt,
                                t_span=ts_fwd,
                                y0=x,
                                t_eval=te_fwd)
            lt_fwd = sol_fwd.t.shape[0]
            y[-1, c_y:c_y + lt_fwd] = sol_fwd.t
            y[:d, c_y:c_y + lt_fwd] = sol_fwd.y
            c_y += lt_fwd

            ini_t = sol_fwd.t[-1]
            x = y[:d, c_y - 1]
            if ini_t == end_t:  # to avoid ini_t == end_t
                break
        if solm.status == 0:
            break
        if solm.status == -1:
            print("Integration step failed")
    print(time() - t1)
    return init_fai, y[:, :c_y], pocr_y[:, :c_pocr]


# When events == True ===========================================
# ===============================================================
n = 4
d = 2 * n + 2
TIME, STEPS = 100, 10000
# variables ================================
Aij = np.ones((n, n))  # coupling matrix
np.fill_diagonal(Aij, 0)
Aij = np.array([[0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0]])
# ==========================================
# constant parameters ======================
B0 = 5.115  # 5.115
r = 1.0
m = 1.0
l = 0.24849
g = 9.81
kth = 34  # 34 # 3
cth = np.log(2)
kfai = 17.75  # 17.75 1 # not too big
cfai = 0.01
epsilon = 5 * np.pi / 180
M = 0.075  # 0.3 # 0.075, for discontinuty
# ==========================================
# constant matrix ==========================
alpha = np.pi / 2 + 2 * np.pi / n * np.arange(n)
sigma = np.zeros(n)
ME = np.zeros(n)
s = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        s[i, j] = r * \
            np.sqrt(2 * (1 - np.cos(alpha[i] - alpha[j])))
rp2 = np.power(r, 2)
lp2 = np.power(l, 2)
A = np.zeros((d, d))
B = np.zeros(d)
# ==========================================


# ===============================================================
# initialization, for n=4 =======================================
init_fai = np.array([-np.pi / 4,
                     np.pi / 4,
                     -np.pi / 4,
                     np.pi / 4])
ptr = 1
# ptr for poincare, for example using fai_1=0 and \dot{fai_1} > 0
fai, y, pocr_y = solution(init_fai, ptr)
for i in range(1, n + 1):
    if y[i, 0] >= np.pi or y[i, 0] <= -np.pi:
        y[i, :] = (y[i, :] % (2 * np.pi)) - ((y[i, :] %
                                              (2 * np.pi)) // np.pi) * (2 * np.pi)
    if pocr_y[i, 0] >= np.pi or pocr_y[i, 0] <= -np.pi:
        pocr_y[i, :] = (pocr_y[i, :] % (2 * np.pi)) - ((pocr_y[i, :] %
                                                        (2 * np.pi)) // np.pi) * (2 * np.pi)
t = y[-1, :]
plt.plot(t, y[0, :], label=r'$\Theta$')
plt.plot(t, y[1, :], label=r'$\phi_1$')
plt.plot(t, y[2, :], label=r'$\phi_2$')
plt.plot(t, y[3, :], label=r'$\phi_3$')
plt.plot(t, y[4, :], label=r'$\phi_3$')
plt.legend()
plt.show()


np.save('./Submission/CHAOS/Multistability/~Metadata/y.npy', y)
np.save('./Submission/CHAOS/Multistability/~Metadata/py.npy', pocr_y)
# ===============================================================
# ===============================================================
