#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import torch
import math


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def getDis(globalW, w):
    sumDis = 0
    w_avg = copy.deepcopy(w)
    for i in w_avg.keys():
        sumDis += torch.norm(w[i] - globalW[i], 2)
    return pow(float(sumDis), 0.5)


def fuzzy_membership(d, r, n=2):
    """
    计算模糊隶属度
    d: 距离
    r: 半径参数
    n: 模糊指数(通常取2)
    """
    return np.exp(-(d ** n) / r)


def fuzzy_entropy(data, m=2, r=0.2):
    """
    计算模糊熵
    替代原来的信息熵，但保持熵的基本性质：
    1. 非负性
    2. 单调性
    3. 极值性
    """
    try:
        N = len(data)
        if N < m + 1:
            return 0
        
        # 数据标准化
        data = np.array(data, dtype=float)
        std = np.std(data)
        if std == 0:
            return 0
        
        r = r * std
        phi_m = 0
        phi_m1 = 0
        
        # 计算模糊熵
        for i in range(N - m):
            count_m = 0
            count_m1 = 0
            
            for j in range(N - m):
                if i == j:
                    continue
                
                # 计算模式距离
                d_m = max([abs(data[i + k] - data[j + k]) for k in range(m)])
                d_m1 = max([abs(data[i + k] - data[j + k]) for k in range(m + 1)])
                
                # 计算模糊隶属度
                count_m += fuzzy_membership(d_m, r)
                count_m1 += fuzzy_membership(d_m1, r)
            
            phi_m += count_m / (N - m - 1)
            phi_m1 += count_m1 / (N - m - 1)
        
        if phi_m == 0:
            return 0
            
        # 返回模糊熵值
        return -np.log(phi_m1 / phi_m)
    except Exception as e:
        print(f"Error in fuzzy_entropy: {str(e)}")
        return 0


def getWk(N_D, s, s_i):
    """
    计算权重
    保持原有的熵权法计算逻辑：
    1. 计算熵值
    2. 计算差异系数
    3. 归一化得到权重
    """
    try:
        if N_D <= 1:  # 至少需要2个样本
            return 1.0 / len(s)
            
        # 计算当前指标的熵值
        entropy = fuzzy_entropy(s_i)
        
        # 计算所有指标的熵值之和
        sum_entropy = 0
        for i in range(len(s)):
            if s[i]:
                sum_entropy += fuzzy_entropy(s[i])
        
        # 计算差异系数 (1 - 熵值)
        diff_coefficient = 1 - entropy
        
        # 所有指标的差异系数之和
        sum_diff_coefficient = len(s) - sum_entropy
        
        if sum_diff_coefficient == 0:
            return 1.0 / len(s)
            
        # 计算权重 (保持原有的权重计算公式)
        weight = diff_coefficient / sum_diff_coefficient
        
        return weight
    except Exception as e:
        print(f"Error in getWk: {str(e)}")
        return 1.0 / len(s)


def getTauI(i, N_D, s):
    """
    计算综合得分
    保持原有的加权求和逻辑
    """
    try:
        sum_score = 0
        sum_weight = 0
        
        # 计算加权和
        for k in range(len(s)):
            if s[k] and len(s[k]) > i:
                weight = getWk(N_D, s, s[k])
                sum_score += weight * s[k][i]
                sum_weight += weight
        
        if sum_weight == 0:
            return 0.5
            
        # 返回归一化后的得分
        return sum_score / sum_weight
    except Exception as e:
        print(f"Error in getTauI: {str(e)}")
        return 0.5


# 根据牛顿冷却法取得当前模型的权重
def getR(t, t0, theta, R0):
    return R0 * pow(math.e, -1 * (theta * (t - t0)))


def normalization(s):
    """
    归一化处理，添加错误处理
    """
    try:
        if not s:
            return 0
            
        res = []
        for k in range(len(s)):
            if not s[k]:
                res.append([])
                continue
                
            current_sum = sum(s[k])
            if current_sum == 0:
                res.append([0] * len(s[k]))
            else:
                res.append([val / current_sum for val in s[k]])
        
        return res
    except Exception as e:
        print(f"Error in normalization: {str(e)}")
        return 0


def getAlpha(kexi, t, t0, theta, R0, i, N_D, s):
    """
    计算最终权重
    保持原有的时间衰减和熵权重组合逻辑
    """
    try:
        if N_D <= 0:
            return 0
            
        # 归一化数据
        s_norm = normalization(s)
        if s_norm == 0:
            return 0
        
        # 计算时间衰减因子
        time_factor = getR(t, t0, theta, R0)
        
        # 计算熵权重
        entropy_weight = getTauI(i, N_D, s_norm)
        
        # 计算最终权重 (保持原有公式)
        alpha = kexi * time_factor * entropy_weight
        
        return max(0.0, min(1.0, alpha))
    except Exception as e:
        print(f"Error in getAlpha: {str(e)}")
        return 0
