#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import torch
import math


# === Functions for BAFL (Fuzzy Entropy) and FedAsync ===

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def getDis(globalW, w): # Used by BAFL and potentially IEWM
    sumDis = 0
    w_avg = copy.deepcopy(w)
    for i in w_avg.keys():
        sumDis += torch.norm(w[i] - globalW[i], 2)
    return pow(float(sumDis), 0.5)


def fuzzy_membership(d, r, n=2):
    """
    计算模糊隶属度 (BAFL specific)
    d: 距离
    r: 半径参数
    n: 模糊指数(通常取2)
    """
    return np.exp(-(d ** n) / r)


def fuzzy_entropy(data, m=5, r=0.5):
    """
    计算模糊熵 (BAFL specific)
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


# Modify getWk to accept fuzzy_r and pass it to fuzzy_entropy
def getWk(N_D, s, s_i, fuzzy_m=5, fuzzy_r=0.5): # Add fuzzy_r parameter
    """
    计算权重 (BAFL specific - uses fuzzy_entropy)
    保持原有的熵权法计算逻辑：
    1. 计算熵值
    2. 计算差异系数
    3. 归一化得到权重
    """
    try:
        if N_D <= 1:  # 至少需要2个样本
            # Return equal weight if only one sample or less
            return 1.0 / len(s) if len(s) > 0 else 1.0 # Avoid division by zero if s is empty

        # 计算当前指标的熵值
        # Pass fuzzy_m and fuzzy_r to fuzzy_entropy
        entropy = fuzzy_entropy(s_i, m=fuzzy_m, r=fuzzy_r)

        # 计算所有指标的熵值之和
        sum_entropy = 0
        valid_metrics_count = 0 # Count metrics that are not empty
        for i in range(len(s)):
            if s[i]: # Check if the metric list is not empty
                # Pass fuzzy_m and fuzzy_r to fuzzy_entropy
                sum_entropy += fuzzy_entropy(s[i], m=fuzzy_m, r=fuzzy_r)
                valid_metrics_count += 1

        if valid_metrics_count == 0: # Handle case where all metric lists in s are empty
             return 1.0 / len(s) if len(s) > 0 else 1.0 # Return equal weight or 1.0 if s is empty

        # 计算差异系数 (1 - 熵值)
        diff_coefficient = 1 - entropy

        # 所有指标的差异系数之和 (use valid_metrics_count)
        sum_diff_coefficient = valid_metrics_count - sum_entropy

        if sum_diff_coefficient == 0:
             # Return equal weight among valid metrics
            return 1.0 / valid_metrics_count if valid_metrics_count > 0 else 1.0

        # 计算权重 (保持原有的权重计算公式)
        weight = diff_coefficient / sum_diff_coefficient

        # Ensure weight is non-negative
        return max(0.0, weight)
    except Exception as e:
        print(f"Error in getWk: {str(e)}")
        # Return equal weight on error
        return 1.0 / len(s) if len(s) > 0 else 1.0


# Modify getTauI to accept fuzzy_r and pass it to getWk
def getTauI(i, N_D, s, fuzzy_m=2, fuzzy_r=0.5): # Add fuzzy_r parameter
    """
    计算综合得分 (BAFL specific - uses getWk with fuzzy_entropy)
    保持原有的加权求和逻辑
    """
    try:
        sum_score = 0
        sum_weight = 0

        # 计算加权和
        for k in range(len(s)):
            # Check if s[k] is valid and has element at index i
            if s[k] and len(s[k]) > i:
                # Pass fuzzy_m and fuzzy_r to getWk
                weight = getWk(N_D, s, s[k], fuzzy_m=fuzzy_m, fuzzy_r=fuzzy_r) # BAFL's getWk
                # Ensure s[k][i] is numeric before multiplication
                try:
                    value = float(s[k][i])
                    sum_score += weight * value
                    sum_weight += weight
                except (ValueError, TypeError):
                    print(f"Warning in getTauI: Non-numeric value encountered at s[{k}][{i}]: {s[k][i]}. Skipping.")
                    continue # Skip this metric for the current node

        if sum_weight == 0:
            print("Warning in getTauI: Sum of weights is zero. Returning default score 0.5.")
            return 0.5

        # 返回归一化后的得分
        # Ensure score is within a reasonable range if needed, e.g., [0, 1] if s[k][i] are normalized
        calculated_score = sum_score / sum_weight
        # return max(0.0, min(1.0, calculated_score)) # Optional: Clamp score if needed
        return calculated_score
    except Exception as e:
        print(f"Error in getTauI: {str(e)}")
        return 0.5 # Return default score on error


# 根据牛顿冷却法取得当前模型的权重 (Used by BAFL and IEWM)
def getR(t, t0, theta, R0):
    return R0 * pow(math.e, -1 * (theta * (t - t0)))


def normalization(s):
    """
    归一化处理，添加错误处理 (Used by BAFL)
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


# Modify getAlpha to accept fuzzy_r and pass it to getTauI
def getAlpha(kexi, t, t0, theta, R0, i, N_D, s, fuzzy_m=2, fuzzy_r=0.5): # Add fuzzy_r parameter
    """
    计算最终权重 (BAFL specific - uses BAFL's normalization and getTauI)
    保持原有的时间衰减和熵权重组合逻辑
    """
    try:
        if N_D <= 0:
            return 0

        # 归一化数据
        s_norm = normalization(s) # BAFL's normalization
        if s_norm == 0:
            return 0

        # 计算时间衰减因子
        time_factor = getR(t, t0, theta, R0)

        # 计算熵权重
        # Pass fuzzy_m and fuzzy_r to getTauI
        entropy_weight = getTauI(i, N_D, s_norm, fuzzy_m=fuzzy_m, fuzzy_r=fuzzy_r) # BAFL's getTauI

        # 计算最终权重 (保持原有公式)
        alpha = kexi * time_factor * entropy_weight

        return max(0.0, min(1.0, alpha))
    except Exception as e:
        print(f"Error in getAlpha: {str(e)}")
        return 0


# FedAvg 聚合函数 (Can be used as a baseline or part of other methods)
def fed_avg_aggregate(global_dict, local_dicts, data_sizes):
    """
    使用 FedAvg 算法聚合本地模型。
    :param global_dict: 当前全局模型的状态字典。
    :param local_dicts: 包含多个本地模型状态字典的列表。
    :param data_sizes: 每个本地模型对应的数据集大小列表。
    :return: 聚合后的全局模型状态字典。
    """
    total_data_size = sum(data_sizes)
    if total_data_size == 0:
        # 如果没有数据，直接返回原始全局模型
        return global_dict

    aggregated_dict = copy.deepcopy(global_dict) # Start with a copy of the global dict

    for k in aggregated_dict.keys():
        # Calculate the weighted sum for the current layer
        weighted_sum = torch.zeros_like(aggregated_dict[k], dtype=torch.float32)
        for i, local_dict in enumerate(local_dicts):
            if k in local_dict: # Ensure the key exists in the local dict
                weighted_sum += local_dict[k] * (data_sizes[i] / total_data_size)
            else:
                # Handle cases where a layer might be missing in a local model (optional, depends on FL setup)
                print(f"Warning: Key '{k}' not found in local model {i}. Using global model value for this layer.")
                # Option 1: Use the global model's value (effectively giving it zero weight from this client)
                # weighted_sum += aggregated_dict[k] * (data_sizes[i] / total_data_size) # This might not be desired
                # Option 2: Skip this client for this layer (adjust total weight accordingly - more complex)
                # Option 3: Raise an error
                # For simplicity, let's assume all models have the same structure and use the global value weighted by its proportion
                # A better approach might be needed depending on the exact FL scenario.
                # Let's stick to the standard FedAvg assumption: all models have the same keys.
                # If a key is missing, it indicates an issue. We'll add the weighted global value for now.
                pass # Or handle appropriately

        aggregated_dict[k] = weighted_sum

    return aggregated_dict


# === Functions for Information Entropy Weight Method (IEWM) ===
# === Copied and adapted from temp.py ===

def getP_iewm(s_k, s_k_i):
    """IEWM specific"""
    if s_k_i == 0:
        return 0
    sum_s_k = 0
    # Ensure s_k is iterable and contains numbers
    try:
        for cur_s_k_i in s_k:
            sum_s_k += cur_s_k_i
        if sum_s_k == 0:
            return 0 # Avoid division by zero
        return s_k_i / sum_s_k
    except TypeError:
        print(f"Error in getP_iewm: s_k is not iterable or contains non-numeric types. s_k: {s_k}")
        return 0


def getEk_iewm(N_D, s_k):
    """IEWM specific - uses standard math.log"""
    if N_D <= 1: # log(1) is 0, causing division by zero
        print("Warning in getEk_iewm: N_D <= 1, returning 0 entropy.")
        return 0
    log_N_D = math.log(N_D)
    if log_N_D == 0: # Should not happen if N_D > 1, but as a safeguard
         print("Warning in getEk_iewm: math.log(N_D) is 0, returning 0 entropy.")
         return 0

    sum_val = 0
    # Ensure s_k has length N_D
    if len(s_k) != N_D:
         print(f"Warning in getEk_iewm: Length of s_k ({len(s_k)}) does not match N_D ({N_D}). Using length of s_k.")
         # Adjust N_D or handle error based on desired logic. Using len(s_k) for now.
         # N_D = len(s_k)
         # if N_D <= 1: return 0
         # log_N_D = math.log(N_D)
         # if log_N_D == 0: return 0
         # Or maybe return an error indicator? For now, proceed with caution.

    for i in range(len(s_k)): # Iterate over actual length of s_k
        # Use the actual value s_k[i]
        p = getP_iewm(s_k, s_k[i])
        if p == 0:
            continue
        # Ensure p is positive for log
        if p < 0:
            print(f"Warning in getEk_iewm: Calculated probability p ({p}) is negative. Skipping.")
            continue
        try:
            sum_val += p * math.log(p)
        except ValueError:
             print(f"Error in getEk_iewm: math.log domain error for p = {p}. Skipping.")
             continue # Skip this term if log(p) is invalid

    # Check log_N_D again before division
    if log_N_D == 0:
        print("Error in getEk_iewm: log_N_D became 0 unexpectedly. Cannot calculate entropy.")
        return 0 # Or handle as appropriate

    return -1.0 * (1 / log_N_D) * sum_val


def getWk_iewm(N_D, s, s_k_list):
    """IEWM specific - uses getEk_iewm"""
    sum_Ek = 0
    num_metrics = len(s) # Number of metric lists in s
    if num_metrics == 0:
        print("Warning in getWk_iewm: Input 's' is empty.")
        return 0

    for k in range(num_metrics):
        # Ensure s[k] is a valid list/iterable for getEk_iewm
        if isinstance(s[k], (list, tuple, np.ndarray)):
             # Check if N_D matches the length of the inner list s[k]
             # if len(s[k]) != N_D:
             #     print(f"Warning in getWk_iewm: Length mismatch for metric {k}. N_D={N_D}, len(s[k])={len(s[k])}")
                 # Decide how to handle: skip, adjust N_D, error?
             sum_Ek += getEk_iewm(N_D, s[k])
        else:
            print(f"Warning in getWk_iewm: Metric {k} in 's' is not a list/array. Skipping.")
            # Adjust num_metrics if skipping?
            num_metrics -= 1 # Decrement count of valid metrics if skipping one
            if num_metrics <= 0: # Avoid division by zero later
                 print("Error in getWk_iewm: No valid metrics found in 's'.")
                 return 0

    # Calculate denominator: num_metrics - sum_Ek
    denominator = num_metrics - sum_Ek
    if denominator == 0:
        # Handle case where all entropies are 1 (or num_metrics equals sum_Ek)
        # Return equal weight or handle as error? Equal weight seems reasonable.
        print("Warning in getWk_iewm: Denominator is zero. Returning equal weight.")
        return 1.0 / num_metrics if num_metrics > 0 else 0

    # Calculate entropy for the specific metric list s_k_list
    Ek_current = getEk_iewm(N_D, s_k_list)

    # Calculate weight
    weight = (1 - Ek_current) / denominator
    return weight


def getTauI_iewm(i, N_D, s):
    """IEWM specific - uses getWk_iewm"""
    sum_val = 0
    if not isinstance(s, (list, tuple)) or len(s) == 0:
        print("Error in getTauI_iewm: Input 's' is not a valid list or is empty.")
        return 0 # Or a default value like 0.5?

    num_metrics = len(s)
    total_weight = 0 # Keep track of total weight for normalization if needed

    for k in range(num_metrics):
        # Ensure s[k] is a list and has element at index i
        if isinstance(s[k], (list, tuple, np.ndarray)) and len(s[k]) > i:
            # Check if N_D matches len(s[k]) - consistency check
            # if len(s[k]) != N_D:
            #     print(f"Warning in getTauI_iewm: Length mismatch for metric {k}. N_D={N_D}, len(s[k])={len(s[k])}")
                # Handle mismatch if necessary

            wk = getWk_iewm(N_D, s, s[k])
            # Ensure s[k][i] is a number
            try:
                value = float(s[k][i]) # Attempt to convert to float
                sum_val += wk * value
                total_weight += wk # Accumulate weight
            except (ValueError, TypeError):
                 print(f"Warning in getTauI_iewm: Value s[{k}][{i}] is not numeric ({s[k][i]}). Skipping.")
                 continue
        else:
            # Handle cases where s[k] is not a list or index i is out of bounds
            print(f"Warning in getTauI_iewm: Metric {k} is invalid or index {i} is out of bounds. Skipping.")
            continue

    # Optional: Normalize by total weight if weights don't sum to 1
    # if total_weight != 0 and total_weight != 1:
    #     print(f"Normalizing TauI_iewm score by total weight: {total_weight}")
    #     return sum_val / total_weight

    return sum_val # Return the weighted sum


def normalization_iewm(s):
    """IEWM specific normalization (sum normalization)"""
    res = []
    if not isinstance(s, (list, tuple)):
        print("Error in normalization_iewm: Input 's' is not a list or tuple.")
        return [] # Return empty list on error

    for k in range(len(s)):
        res.append([])
        if not isinstance(s[k], (list, tuple, np.ndarray)):
             print(f"Warning in normalization_iewm: Element {k} in 's' is not a list/array. Appending empty list.")
             continue # Skip to next element

        # Calculate sum of the current sublist s[k]
        try:
            current_sum = sum(s[k])
        except TypeError:
             print(f"Warning in normalization_iewm: Element {k} contains non-numeric types. Cannot sum. Appending empty list.")
             continue # Skip to next element

        if current_sum == 0:
             # If sum is 0, append list of zeros or handle as needed
             # Appending list of zeros to maintain structure
             res[k] = [0.0] * len(s[k])
             # print(f"Warning in normalization_iewm: Sum of element {k} is zero.")
             continue

        # Normalize: divide each element by the sum
        try:
            res[k] = [float(val) / current_sum for val in s[k]]
        except (TypeError, ValueError):
             print(f"Warning in normalization_iewm: Element {k} contains non-numeric types. Could not normalize. Appending empty list.")
             res[k] = [] # Reset to empty on error during division
             continue

    # Check if the outer list 'res' is empty, which might happen if 's' was empty or all elements failed
    if not res:
        print("Warning in normalization_iewm: Resulting normalized list is empty.")
        # Decide return value: empty list, 0, None? Returning empty list for now.
        return []

    # Check if normalization resulted in lists of different lengths if that's unexpected
    # ...

    # The original code returned 0 if the *first* sublist sum was 0.
    # Returning the list `res` seems more appropriate.
    # Let's refine the check for the first sublist sum being zero if that specific behavior is needed.
    # if len(s) > 0 and isinstance(s[0], (list, tuple, np.ndarray)) and sum(s[0]) == 0:
    #      print("Warning: Sum of the first sublist is zero in normalization_iewm.")
         # return 0 # Original behavior - might be problematic

    return res


def getAlpha_iewm(kexi, t, t0, theta, R0, i, N_D, s):
    """IEWM specific alpha calculation"""
    # 归一化解决时间戳数值过大导致的熵权过小的问题。
    s_normalized = normalization_iewm(s)
    # Check if normalization failed (returned empty list or other indicator)
    if not s_normalized: # Assuming empty list indicates failure based on implementation
         print("Error in getAlpha_iewm: Normalization failed.")
         return 0 # Return 0 alpha on normalization error

    # Calculate time decay factor
    time_factor = getR(t, t0, theta, R0)

    # Calculate entropy weight using IEWM's TauI
    entropy_weight = getTauI_iewm(i, N_D, s_normalized)

    # Calculate final alpha
    alpha = kexi * time_factor * entropy_weight

    # Ensure alpha is within [0, 1] bounds (optional, depends on requirements)
    alpha = max(0.0, min(1.0, alpha))

    return alpha
