# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 22:35:08 2025

@author: ericd
"""

import numpy as np

# 定義參數
h = 0.1
x = np.arange(0, 1.1, h)
n = len(x)
y1 = np.zeros(n)
z1 = np.zeros(n)
y2 = np.zeros(n)
z2 = np.zeros(n)

# 初始條件
y1[0] = 1.0  # y1(0) = 1
z1[0] = 0.0  # y1'(0) = 0
y2[0] = 0.0  # y2(0) = 0
z2[0] = 1.0  # y2'(0) = 1

# 定義微分方程
def f1(x, y, z):  # 第一個初值問題
    return np.array([z, -(x+1)*z + 2*y + (1-x**2)*np.exp(-x)])

def f2(x, y, z):  # 第二個初值問題
    return np.array([z, -(x+1)*z + 2*y])

# RK4 迭代
for n in range(n-1):
    # 第一個問題 (y1, z1)
    Y1n = np.array([y1[n], z1[n]])
    k1_1 = f1(x[n], y1[n], z1[n])
    k1_2 = f1(x[n] + h/2, y1[n] + h/2*k1_1[0], z1[n] + h/2*k1_1[1])
    k1_3 = f1(x[n] + h/2, y1[n] + h/2*k1_2[0], z1[n] + h/2*k1_2[1])
    k1_4 = f1(x[n] + h, y1[n] + h*k1_3[0], z1[n] + h*k1_3[1])
    Y1n1 = Y1n + h/6 * (k1_1 + 2*k1_2 + 2*k1_3 + k1_4)
    y1[n+1] = Y1n1[0]
    z1[n+1] = Y1n1[1]

    # 第二個問題 (y2, z2)
    Y2n = np.array(y2[n], z2[n])
    k2_1 = f2(x[n], y2[n], z2[n])
    k2_2 = f2(x[n] + h/2, y2[n] + h/2*k2_1[0], z2[n] + h/2*k2_1[1])
    k2_3 = f2(x[n] + h/2, y2[n] + h/2*k2_2[0], z2[n] + h/2*k2_2[1])
    k2_4 = f2(x[n] + h, y2[n] + h*k2_3[0], z2[n] + h*k2_3[1])
    Y2n1 = Y2n + h/6 * (k2_1 + 2*k2_2 + 2*k2_3 + k2_4)
    y2[n+1] = Y2n1[0]
    z2[n+1] = Y2n1[1]

# 計算 c
y1_1 = y1[-1]
y2_1 = y2[-1]
if abs(y2_1) < 1e-10:
    print("警告：y2(1) 接近零，需調整 y2'(0) 並重新計算")
else:
    c = (2.0 - y1_1) / y2_1
    print(f"y1(1) = {y1_1:.6f}, y2(1) = {y2_1:.6f}, c = {c:.6f}")

# 最終解
y = y1 + c * y2

# 輸出結果
print("\n最終解：")
print("x\t\ty(x)")
for i in range(11):
    print(f"{x[i]:.1f}\t\t{y[i]:.6f}")

# 驗證邊界條件
print(f"\n驗證：y(0) = {y[0]:.6f}, y(1) = {y[-1]:.6f}")