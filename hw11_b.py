# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 22:41:08 2025

@author: ericd
"""

import numpy as np

# 定義參數
h = 0.1
n = 9  # 內部節點數
x = np.arange(0, 1.1, h)  # x = [0, 0.1, ..., 1]
y = np.zeros(n+2)  # 包含邊界點
y[0] = 1.0  # y(0) = 1
y[n+1] = 2.0  # y(1) = 2

# 定義係數函數
def p(x):
    return -(x + 1)

def q(x):
    return 2.0

def r(x):
    return (1 - x**2) * np.exp(-x)

# 構建矩陣 [A] 和向量 {F}
A = np.zeros((n, n))
F = np.zeros(n)

for i in range(n):
    xi = x[i+1]  # x_1 到 x_9
    pi = p(xi)
    qi = q(xi)
    ri = r(xi)
    
    # 主對角線
    A[i, i] = 2 + h**2 * qi  # 2 + h^2 * 2 = 2.02
    
    # 下對角線
    if i > 0:
        A[i, i-1] = -(1 + h/2 * pi)  # -(1 + 0.05 * (-(x_i + 1)))
    
    # 上對角線
    if i < n-1:
        A[i, i+1] = -(1 - h/2 * pi)  # -(1 - 0.05 * (-(x_i + 1)))
    
    # 右端向量
    F[i] = -h**2 * ri
    if i == 0:
        F[i] += (1 + h/2 * pi) * y[0]  # 邊界條件 y_0 = 1
    if i == n-1:
        F[i] += (1 - h/2 * pi) * y[n+1]  # 邊界條件 y_{10} = 2

# 求解線性方程組
y[1:n+1] = np.linalg.solve(A, F)

# 輸出結果
print("有限差分法結果：")
print("x\t\ty(x)")
for i in range(n+2):
    print(f"{x[i]:.1f}\t\t{y[i]:.6f}")

# 驗證邊界條件
print(f"\n驗證：y(0) = {y[0]:.6f}, y(1) = {y[-1]:.6f}")

# 輸出矩陣 A（供檢查）
print("\n矩陣 A：")
print(A)