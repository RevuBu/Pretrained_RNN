import numpy as np


def minjerk(H1, H2, t, n):
    T = np.linspace(0, t, n)
    H = np.zeros((n, 2))
    Hd = np.zeros((n, 2))
    Hdd = np.zeros((n, 2))
    for i in range(n):
        tau = T[i] / t
        H[i, 0] = H1[0] + ((H1[0] - H2[0]) * (15 * (tau**4) - (6 * tau**5) - (10 * tau**3)))
        H[i, 1] = H1[1] + ((H1[1] - H2[1]) * (15 * (tau**4) - (6 * tau**5) - (10 * tau**3)))
        Hd[i, 0] = (H1[0] - H2[0]) * (-30 * T[i]**4 / t**5 + 60 * T[i]**3 / t**4 - 30 * T[i]**2 / t**3)
        Hd[i, 1] = (H1[1] - H2[1]) * (-30 * T[i]**4 / t**5 + 60 * T[i]**3 / t**4 - 30 * T[i]**2 / t**3)
        Hdd[i, 0] = (H1[0] - H2[0]) * (-120 * T[i]**3 / t**5 + 180 * T[i]**2 / t**4 - 60 * T[i] / t**3)
        Hdd[i, 1] = (H1[1] - H2[1]) * (-120 * T[i]**3 / t**5 + 180 * T[i]**2 / t**4 - 60 * T[i] / t**3)
    return T, H, Hd, Hdd


def comMiniJerk(start_postion, goal, move_dur, move_ntps):
    traj_list = []
    vel_list = []
    n_batch = start_postion.shape[0]
    for i in range(n_batch):
        T, H, Hd, Hdd = minjerk(start_postion[i, :], goal[i, :], move_dur[i], int(move_ntps[i]))
        traj_list.append(H)
        vel_list.append(Hd)
    return traj_list, vel_list


def joints_to_hand(A, aparams):
    l1 = aparams["l1"]
    l2 = aparams["l2"]
    n = np.shape(A)[0]
    E = np.zeros((n, 2))
    H = np.zeros((n, 2))
    for i in range(n):
        E[i, 0] = l1 * np.cos(A[i, 0])
        E[i, 1] = l1 * np.sin(A[i, 0])
        H[i, 0] = E[i, 0] + (l2 * np.cos(A[i, 0] + A[i, 1]))
        H[i, 1] = E[i, 1] + (l2 * np.sin(A[i, 0] + A[i, 1]))
    return H, E


def hand_to_joints(H, aparams):
    l1 = aparams["l1"]
    l2 = aparams["l2"]
    n = np.shape(H)[0]
    A = np.zeros((n, 2))
    for i in range(n):
        A[i, 1] = np.arccos(
            ((H[i, 0] * H[i, 0]) + (H[i, 1] * H[i, 1]) - (l1 * l1) - (l2 * l2))
            / (2.0 * l1 * l2)
        )
        A[i, 0] = np.arctan(H[i, 1] / H[i, 0]) - np.arctan(
            (l2 * np.sin(A[i, 1])) / (l1 + (l2 * np.cos(A[i, 1])))
        )
        if A[i, 0] < 0:
            A[i, 0] = A[i, 0] + np.pi
        elif A[i, 0] > np.pi:
            A[i, 0] = A[i, 0] - np.pi
    return A


def computeTheta21(target):
    theta1 = (target[0] - 1) * 60
    theta2 = (target[1] - 1) * 60
    if theta2 - theta1 < 0:
        theta2 = theta2 + 360
    theta = theta1 + 180 - (180 - (theta2 - theta1)) / 2
    theta21 = np.mod(theta, 360)
    return theta21
