# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 22:42:13 2025

@author: ericd
"""

import numpy as np
from scipy.integrate import quad

# 定義參數
h = 0.1
x = np.arange(0, 1.1, h)
n_basis = 2  # 基函數數量
l = 1.0

# 定義基函數
def phi(i, x):
    return np.sin(i * np.pi * x / l)

def dphi_dx(i, x):
    return i * np.pi / l * np.cos(i * np.pi * x / l)

# 定義 F(x)
def F(x):
    return x + 1 + (1 - x**2) * np.exp(-x)

# 構建矩陣 [A] 和向量 {b}
A = np.zeros((n_basis, n_basis))
b = np.zeros(n_basis)

for i in range(n_basis):
    for j in range(n_basis):
        # 計算 a_ij = ∫[0,1] (dphi_i/dx * dphi_j/dx + 2 * phi_i * phi_j) dx
        integrand = lambda x: dphi_dx(i+1, x) * dphi_dx(j+1, x) + 2 * phi(i+1, x) * phi(j+1, x)
        A[i, j] = quad(integrand, 0, 1)[0]
    
    # 計算 b_i = ∫[0,1] F(x) * phi_i dx
    integrand_b = lambda x: F(x) * phi(i+1, x)
    b[i] = quad(integrand_b, 0, 1)[0]

# 求解 c_i
c = np.linalg.solve(A, b)

# 計算 y(x) = y_1(x) + y_2(x)
y = np.zeros(len(x))
for i in range(len(x)):
    y_1 = 1 + x[i]  # y_1(x) = 1 + x
    y_2 = sum(c[j] * phi(j+1, x[i]) for j in range(n_basis))  # y_2(x) = Σ c_i * sin(iπx)
    y[i] = y_1 + y_2

# 輸出結果
print("變分法結果：")
print("x\t\ty(x)")
for i in range(len(x)):
    print(f"{x[i]:.1f}\t\t{y[i]:.6f}")

# 驗證邊界條件
print(f"\n驗證：y(0) = {y[0]:.6f}, y(1) = {y[-1]:.6f}")

# 輸出 A 和 c
print("\n矩陣 A：")
print(A)
print("\n係數 c：")
print(c)