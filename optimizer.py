import numpy as np
import matplotlib.pyplot as plt

# 1. 목적 함수 및 그래디언트 정의 (극단적인 타원형 계곡)
# w1 방향은 매우 완만(0.1), w2 방향은 매우 가파름(5.0)
def f(w1, w2):
    return 0.1 * w1**2 + 5.0 * w2**2

def grad_f(w1, w2):
    return np.array([0.2 * w1, 10.0 * w2])

# 2. 하이퍼파라미터 설정
start_point = np.array([-8.0, 4.0])
iterations = 45 # 최적점 부근의 움직임을 보기 위해 횟수 지정

# Gradient Descent 설정
lr_gd = 0.19  # w2 방향으로 심하게 진동하도록 한계치에 가깝게 설정
path_gd = [start_point]
w_gd = start_point.copy()

# RMSprop 설정
lr_rms = 0.8  # 빠른 수렴과 최적점 부근의 진동(단점)을 유발하기 위한 학습률
beta = 0.9
epsilon = 1e-8
path_rms = [start_point]
w_rms = start_point.copy()
sq_grad_avg = np.zeros(2)

# 3. 최적화 진행
for _ in range(iterations):
    # Gradient Descent 업데이트
    g_gd = grad_f(w_gd[0], w_gd[1])
    w_gd = w_gd - lr_gd * g_gd
    path_gd.append(w_gd)
    
    # RMSprop 업데이트
    g_rms = grad_f(w_rms[0], w_rms[1])
    sq_grad_avg = beta * sq_grad_avg + (1 - beta) * (g_rms**2)
    w_rms = w_rms - (lr_rms / (np.sqrt(sq_grad_avg) + epsilon)) * g_rms
    path_rms.append(w_rms)

path_gd = np.array(path_gd)
path_rms = np.array(path_rms)

# 4. 시각화 (Plotting)
w1_range = np.linspace(-10, 10, 400)
w2_range = np.linspace(-5, 5, 400)
W1, W2 = np.meshgrid(w1_range, w2_range)
Z = f(W1, W2)

plt.figure(figsize=(10, 7))

# 등고선 그리기 (간격을 촘촘하게 하여 계곡 모양 강조)
levels = np.logspace(-1, 3, 20)
plt.contour(W1, W2, Z, levels=levels, colors='indigo', alpha=0.3)

# 궤적 그리기
plt.plot(path_gd[:, 0], path_gd[:, 1], 'o-', markersize=4, label='Gradient Descent (Severe Oscillation)', color='tab:blue')
plt.plot(path_rms[:, 0], path_rms[:, 1], 's-', markersize=4, label='RMSprop (Fast, but Jitters at end)', color='tab:green')

# 출발점 및 최적점 표시
plt.plot(start_point[0], start_point[1], 'bD', markersize=8, label='Start Point')
plt.plot(0, 0, 'r*', markersize=15, label='Optimal Point (0,0)')

plt.title('RMSprop Pros & Cons: Trajectory Comparison', fontsize=14, pad=15)
plt.xlabel('w1 (Flat direction)', fontsize=12)
plt.ylabel('w2 (Steep direction)', fontsize=12)
plt.xlim(-10, 10)
plt.ylim(-5, 5)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()