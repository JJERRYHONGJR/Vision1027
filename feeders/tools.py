import random

import numpy as np


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])  # xuanzhuan juzhen

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]  # pingyi bianhuan
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    # input: C,T,V,M 偏移其中一段
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift

###################################################################################
def random_shift(data_numpy, max_shift=5):
    # 随机在 T 维度上偏移，且允许在每个轴上有不同的偏移值
    C, T, V, M = data_numpy.shape
    shift = np.random.randint(-max_shift, max_shift + 1)
    data_shift = np.zeros_like(data_numpy)
    
    # 保证不会移出数据的有效范围
    if shift > 0:
        data_shift[:, shift:, :, :] = data_numpy[:, :-shift, :, :]
    elif shift < 0:
        data_shift[:, :shift, :, :] = data_numpy[:, -shift:, :, :]
    else:
        data_shift = data_numpy

    return data_shift

def add_random_noise(data_numpy, noise_level=0.01):
    # 随机添加高斯噪声
    noise = np.random.normal(0, noise_level, data_numpy.shape)
    data_numpy += noise
    return data_numpy



def random_rotation(data_numpy, max_angle=10,angle_candidate=[-10., -5., 0., 5., 10.]):
    # 随机在X-Y平面上旋转，角度范围为 [-max_angle, max_angle]
    C, T, V, M = data_numpy.shape
    data_numpy = data_numpy.copy()  # 创建可修改的副本
    angle = np.random.uniform(-max_angle, max_angle) * np.pi / 180  # 角度转弧度

    # 构造旋转矩阵
    cos_val, sin_val = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_val, -sin_val], [sin_val, cos_val]])

    # 只对 X 和 Y 坐标进行旋转 (假设前两个维度是 X 和 Y)
  
    for t in range(T):
        angle = random.choice(angle_candidate)
        rad = np.deg2rad(angle)
        rotation_matrix = np.array([[np.cos(rad), -np.sin(rad)],
                                    [np.sin(rad), np.cos(rad)]])
        xy = data_numpy[:2, t, :, :]
        data_numpy[:2, t, :, :] = np.dot(rotation_matrix, xy.reshape(2, -1)).reshape(2, V, M)
   
    return data_numpy




def random_scaling(data_numpy, scale_range=(0.9, 1.1)):
    # 随机缩放比例
    scale_factor = np.random.uniform(*scale_range)
    data_numpy[:2, :, :, :] *= scale_factor  # 只对 X 和 Y 进行缩放
    return data_numpy




def random_temporal_crop(data_numpy, size, auto_pad=True):
    # 对序列进行时间维度上的随机裁剪
    C, T, V, M = data_numpy.shape
    if T <= size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        start = np.random.randint(0, T - size + 1)
        return data_numpy[:, start:start + size, :, :]




def random_frame_drop(data_numpy, drop_prob=0.1):
    # 以一定概率随机丢弃时间帧
    C, T, V, M = data_numpy.shape
    drop_mask = np.random.rand(T) > drop_prob
    return data_numpy[:, drop_mask, :, :]



def combined_augmentation(data_numpy, rotation=True, noise=True, scaling=True, frame_drop=True):
    if rotation:
        data_numpy = random_rotation(data_numpy)
    if noise:
        data_numpy = add_random_noise(data_numpy)
    if scaling:
        data_numpy = random_scaling(data_numpy)
    if frame_drop:
        data_numpy = random_frame_drop(data_numpy)
    return data_numpy
############################################################################################




def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy
