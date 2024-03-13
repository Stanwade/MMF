import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

# 时间点
time_points = np.linspace(0, 999, num=100)
# 变量取值范围
x_values = np.linspace(-3, 3, num=400)

# 初始化热力图数据数组
heatmap_data = np.zeros((len(time_points), len(x_values)))


betas = np.linspace(0.001, 0.04, num=100)
alphas = 1 - betas

alpha_bars = np.empty_like(alphas)
product = 1

for i in range(len(alphas)):
    product *= alphas[i]
    alpha_bars[i] = product

exp = 1.35

x1 = 1
x2 = -x1
var = 0.2

data_pdf1 = stats.norm.pdf(x_values, x1, var)
data_pdf2 = stats.norm.pdf(x_values, x2, var)
noise_pdf = stats.norm.pdf(x_values, 0, 1)

for i, t in enumerate(time_points):
    
    t = int(t / 10)
    print(f'iter {i}; time {t}')
    x1_t = math.sqrt(alpha_bars[t]) * x1
    x2_t = math.sqrt(alpha_bars[t]) * x2
    var = math.sqrt(1 - alpha_bars[t])
    
    pdf1 = stats.norm.pdf(x_values, x1_t, var)
    pdf2 = stats.norm.pdf(x_values, x2_t, var)
    
    # 计算每个时间点的混合PDF
    mixed_pdf = pdf1 + pdf2
    # 填充热力图数据
    heatmap_data[i] = mixed_pdf
    heatmap_data[i] = heatmap_data[i] / np.max(heatmap_data[i])

# 绘制热力图
plt.figure(figsize=(20, 8))
plt.imshow(heatmap_data.transpose(), extent=[time_points.min(), time_points.max(), x_values.min(), x_values.max()], aspect='auto', origin='lower',vmax=1)
plt.colorbar(label='PDF value')
plt.xlabel('Value')
plt.ylabel('Time')
plt.title('Heatmap of Distribution Transition from Bimodal to Standard Normal')
plt.show()
