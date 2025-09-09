import os
import numpy as np
import matplotlib.pyplot as plt
from openmm.app import PDBFile
from openmm.unit import dalton, nanometer
import scienceplots

plt.style.use(['science', 'grid'])

plt.rcParams.update({
    'font.size': 14,           # 폰트 크기
    'axes.titlesize': 22,      # 축 제목 폰트 크기
    'axes.labelsize': 22,      # 축 레이블 폰트 크기
    'xtick.labelsize': 22,     # x축 tick 폰트 크기
    'ytick.labelsize': 22,     # y축 tick 폰트 크기
    'legend.fontsize': 14,     # 범례 폰트 크기
    'lines.markersize': 10,     # 마커 크기
    'lines.linewidth': 2,       # 선 두께
    'xtick.major.size': 10,  # x축 주 tick 길이
    'ytick.major.size': 10,  # y축 주 tick 길이
    'xtick.minor.size': 5,   # x축 부 tick 길이
    'ytick.minor.size': 5,   # y축 부 tick 길이
})

# ----------------------------
# 입력 경로 & 레퍼런스 준비
# ----------------------------
paths = [
    '../paths/g4/250810_g4Na_350K_20ps/scale/',
    '../paths/g4/250810_g4Na_350K_20ps_2/scale/',
    '../paths/g4/250810_g4Na_350K_20ps_3/scale/'
]

state_A_pdb = PDBFile('../data/g4/143d_Na.pdb')
state_B_pdb = PDBFile('../data/g4/1kf1_K.pdb')

# (옵션) 질량 배열 – 필요시 mass-weighted로 확장 가능
masses = np.array([atom.element.mass.value_in_unit(dalton) for atom in state_A_pdb.topology.atoms()])

RA = state_A_pdb.getPositions(asNumpy=True).value_in_unit(nanometer)  # (N,3)
RB = state_B_pdb.getPositions(asNumpy=True).value_in_unit(nanometer)  # (N,3)

# ----------------------------
# Kabsch RMSD
# ----------------------------
def kabsch_rmsd(P, Q):
    """
    RMSD between P and Q after optimal rotation+translation (Kabsch).
    P, Q: (N, 3) numpy arrays, same ordering of atoms.
    """
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    U = V @ np.diag([1.0, 1.0, d]) @ Wt
    Pr = Pc @ U
    diff2 = ((Pr - Qc) ** 2).sum()
    return np.sqrt(diff2 / P.shape[0])

# ----------------------------
# Trajectory 한 개(positions.npy) -> (rmsdA[], rmsdB[])로 변환
# ----------------------------
def calculate_rmsd_path(positions, RA, RB):
    """
    positions: (T, N, 3) in nm
    returns: rmsdA (T,), rmsdB (T,)
    """
    T = positions.shape[0]
    rmsdA = np.empty(T, dtype=np.float64)
    rmsdB = np.empty(T, dtype=np.float64)
    for t in range(T):
        P = positions[t]  # (N,3)
        rmsdA[t] = kabsch_rmsd(P, RA)
        rmsdB[t] = kabsch_rmsd(P, RB)
    return rmsdA, rmsdB

# ----------------------------
# Plot
# ----------------------------
fig, ax = plt.subplots(figsize=(5.0, 4.5))
colors = plt.cm.tab10.colors  # path별 색상

for p_idx, p in enumerate(paths):
    color = colors[p_idx % len(colors)]
    # 각 path 밑의 positions 디렉토리에서 0..7.npy 로드
    pos_dir = os.path.join(p, 'positions')
    for i in range(8):
        npy_path = os.path.join(pos_dir, f'{i}.npy')
        if not os.path.isfile(npy_path):
            print(f"[WARN] 파일 없음: {npy_path}  -> 건너뜀")
            continue

        positions = np.load(npy_path)  # (T,N,3), 단위 nm 가정
        rmsdA, rmsdB = calculate_rmsd_path(positions, RA, RB)

        # path 곡선
        ax.plot(rmsdA, rmsdB, lw=1, color=color, alpha=0.9)

        # 시작/끝 점: 흰색 내부 + 검정 테두리 동그라미
        ax.scatter([rmsdA[0]], [rmsdB[0]], s=50, facecolor='white', edgecolor='black', zorder=5)
        ax.scatter([rmsdA[-1]], [rmsdB[-1]], s=50, facecolor='white', edgecolor='black', zorder=5)

# 축/레이블/스타일
ax.set_xlabel('RMSD, 143D(Na) (nm)')
ax.set_ylabel('RMSD, 1KF1(K) (nm)')
ax.set_xlim(0.0, 2.5)
ax.set_ylim(0.0, 2.5)
ax.set_aspect('equal', adjustable='box')  # x,y 동일 스케일
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

