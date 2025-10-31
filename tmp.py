import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# ------------------------------------------
# 1️⃣ 파일 경로
# ------------------------------------------
disp_path = "0000000000_disp.npy"
rgb_path  = "0000000000.jpg"

## ------------------------------------------
# 2️⃣ disp + RGB 불러오기
# ------------------------------------------
disp = np.load(disp_path)
if disp.ndim == 3:
    disp = disp.squeeze()
elif disp.ndim == 4:
    disp = disp[0, 0]
H, W = disp.shape

rgb = np.array(Image.open(rgb_path).convert("RGB")) / 255.0
if rgb.shape[:2] != (H, W):
    rgb = np.array(Image.fromarray((rgb*255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255.0
    print(f"Resized RGB to {rgb.shape}")

# ------------------------------------------
# 3️⃣ Disparity → Depth 변환
# ------------------------------------------
depth = 1.0 / (disp + 1e-6)
depth = depth / np.percentile(depth, 99) * 10.0  # Normalize (0~10m)

# ------------------------------------------
# 4️⃣ 픽셀 좌표 → 3D 변환
# ------------------------------------------
fx, fy = 1.0, 1.0
cx, cy = W / 2, H / 2
xmap, ymap = np.meshgrid(np.arange(W), np.arange(H))
X = (xmap - cx) / fx * depth
Y = (ymap - cy) / fy * depth
Z = depth

# ------------------------------------------
# 5️⃣ 법선 계산
# ------------------------------------------
def compute_normals(depth):
    dzdx = np.gradient(depth, axis=1)
    dzdy = np.gradient(depth, axis=0)
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(depth)
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
    return nx / norm, ny / norm, nz / norm

Nx, Ny, Nz = compute_normals(depth)

# ------------------------------------------
# 6️⃣ 여러 평면 정의 (z≈값)
# ------------------------------------------
planes = np.linspace(1.0, 9.0, 5)  # 1, 3, 5, 7, 9 m
tolerance = 0.2
colors = plt.cm.jet(np.linspace(0, 1, len(planes)))

# ------------------------------------------
# 7️⃣ 각 평면별 법선 벡터 시각화 + 평균 계산
# ------------------------------------------
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Normals of Pixels in Each Depth Plane")

print("\n=== 평균 법선 벡터 (Mean Surface Normal per Plane) ===\n")

for z0, c in zip(planes, colors):
    mask = (Z > z0 - tolerance) & (Z < z0 + tolerance)
    if mask.sum() < 50:
        continue

    # 해당 평면 픽셀 좌표
    x_plane = X[mask]
    y_plane = Y[mask]
    z_plane = Z[mask]
    nx_plane = Nx[mask]
    ny_plane = Ny[mask]
    nz_plane = Nz[mask]

    # 평균 법선 계산
    n_mean = np.array([nx_plane.mean(), ny_plane.mean(), nz_plane.mean()])
    n_mean /= np.linalg.norm(n_mean) + 1e-8

    # 터미널 출력
    print(f"z≈{z0:.1f}m plane → mean normal: ({n_mean[0]:.3f}, {n_mean[1]:.3f}, {n_mean[2]:.3f}), N={mask.sum()} pixels")

    # 샘플링 (너무 많으면 일부만)
    step = max(1, len(x_plane)//300)
    x_plane, y_plane, z_plane = x_plane[::step], y_plane[::step], z_plane[::step]
    nx_plane, ny_plane, nz_plane = nx_plane[::step], ny_plane[::step], nz_plane[::step]

    # 화살표 시각화
    ax.quiver(x_plane, y_plane, z_plane,
              nx_plane, ny_plane, nz_plane,
              length=0.4, color=c, normalize=True, label=f"z≈{z0:.1f}m")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z (Depth)")
ax.view_init(elev=25, azim=-60)
ax.legend()
plt.tight_layout()
plt.show()