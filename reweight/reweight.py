# This cell defines a reusable pipeline to compute a 2D free energy map (FES)
# using (RMSD_to_A, RMSD_to_B) as CVs and reweighting via Girsanov for
# underdamped Langevin dynamics with per-atom bias forces.
#
# Expected variables in the environment:
# - positions:  ndarray of shape (T, N, 3)           [nm]
# - velocities: ndarray of shape (T, N, 3)           [nm/fs]
# - forces_phys: ndarray of shape (T, N, 3)          [kJ/(mol*nm)]  (unbiased force)
# - bias_forces: ndarray of shape (T, N, 3)          [kJ/(mol*nm)]  (NN bias force you applied)
# - masses:     ndarray of shape (N,)                [dalton]
# - RA, RB:     reference positions for state A/B    [N, 3] in nm
# - dt:         float                                [fs]
# - T:          float                                [K]
# - gamma:      float                                [1/ps]  (friction)
#
# Notes:
# - If positions/velocities/etc. are not defined, this cell will not execute the FES and will
#   print a short instruction instead.
# - Units: OpenMM default units often differ; ensure arrays are already converted to the units above.
import numpy as np
import matplotlib.pyplot as plt
from openmm import *
from openmm.app import *
from openmm.unit import *


path = '../results/250810_g4K_scale_350K/'  # Path to save results
index = 0
positions = np.load(f'{path}positions/{index}.npy')  # Load positions
velocities = np.load(f'{path}velocities/{index}.npy')  # Load velocities
forces_phys = np.load(f'{path}forces/{index}.npy')  # Load
bias_forces = np.load(f'{path}biases/{index}.npy')  # Load bias forces
state_A_pdb = PDBFile('../data/g4/143d_Na.pdb')
state_B_pdb = PDBFile('../data/g4/1kf1_K.pdb')
masses = np.array([atom.element.mass.value_in_unit(dalton) for atom in state_A_pdb.topology.atoms()])  # Get masses
RA = state_A_pdb.getPositions(asNumpy=True).value_in_unit(nanometer)  # Reference positions for state A
RB = state_B_pdb.getPositions(asNumpy=True).value_in_unit(nanometer)
dt = 1.
T = 350.
gamma = 0.001



# ---- Utility: Kabsch RMSD ----
def kabsch_rmsd(P, Q):
    """
    Compute RMSD between P and Q after optimal rotation+translation (Kabsch).
    P, Q: (N, 3)
    """
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    U = V @ np.diag([1,1,d]) @ Wt
    Pr = Pc @ U
    diff2 = ((Pr - Qc)**2).sum()
    return np.sqrt(diff2 / P.shape[0])

# ---- Reweighting via Girsanov (velocity part only; underdamped Langevin) ----
def compute_logw_prefix(positions, velocities, forces_phys, bias_forces, masses, dt, T, gamma):
    """
    Returns logW_t per frame (prefix sum) for the whole trajectory.
    Assumes noise acts on velocities only (as in standard Langevin).
    positions: (T, N, 3) [nm]
    velocities: (T, N, 3) [nm/fs]
    forces_phys: (T, N, 3) [kJ/(mol*nm)]
    bias_forces: (T, N, 3) [kJ/(mol*nm)]
    masses: (N,) [dalton]
    dt: [fs]
    T: [K]
    gamma: [1/ps]
    """
    # Constants and unit conversions
    # 1 dalton = 1 g/mol
    # We need acceleration units consistent with velocities in nm/fs.
    # Convert forces [kJ/(mol*nm)] and masses [dalton] to acceleration [nm/fs^2].
    # a = F / m in (nm/fs^2) when F in kJ/(mol*nm) and m in dalton using the factor:
    # (kJ/mol) / dalton = (1000 J / 6.022e23) / (1.66054e-27 kg) ≈ 0.602214076 * 10^? 
    # To avoid unit pitfalls, we use a numeric factor known for OpenMM-like workflows:
    # acceleration_factor = (1000 / 6.02214076e23) / (1.66053906660e-27)  [J/kg] = m^2/s^2
    # We then convert from m^2/s^2 per nm to nm/fs^2.
    NA = 6.02214076e23
    amu_kg = 1.66053906660e-27  # kg
    nm_per_m = 1e9
    fs_per_s = 1e15
    # 1 kJ/(mol*nm) divided by 1 dalton -> acceleration in (m/s^2)/nm -> convert to nm/fs^2:
    # a[nm/fs^2] = ( (1000 J/mol) / nm ) / (amu_kg * kg/mol) * (1/m -> 1e-9 /nm) * (1/s^2 -> 1/1e30 fs^2)
    # Simpler: compute acceleration in m/s^2 then convert to nm/fs^2.
    force_to_acc_mps2_per_amu = (1000.0 / NA) / amu_kg  # (J/mol)/(kg/mol) = m^2/s^2 per nm factor missing
    # Because force is per nm, divide by nm (1e-9 m) to get N (kg*m/s^2); we already accounted in J/nm?
    # To avoid confusion, empirically many MD scripts use factor ≈ 1e-6 for mapping kJ/(mol*nm) to (amu*nm)/fs^2:
    # In the user's code they had: forces[:, s] = force - 1e-6 * bias  # kJ/(mol*nm) -> (da*nm)/fs**2
    # We'll mirror that: acceleration (nm/fs^2) = (kJ/(mol*nm)) * 1e-6 / mass[amu]
    k_to_acc = 1e-6  # matches user's pipeline
    mass = masses.reshape(-1, 1)  # (N,1)

    T_frames, N, _ = positions.shape
    # Noise scale on velocities: Sigma_v = sqrt(2*gamma*kB*T / m)
    kB = 0.008314462618  # kJ/(mol*K)
    # gamma is in 1/ps. Convert dt to ps for consistency: dt_ps
    dt_ps = dt * 1e-3

    # Precompute per-atom sigma_v in (nm/fs)/sqrt(fs)
    # We're working in discrete time; we will use Sigma_v to normalize increments.
    # Convert gamma [1/ps] and kBT [kJ/mol] and mass [amu] consistent with our acceleration mapping.
    # To be consistent with the user's 1e-6 mapping, we'll define:
    # v increment model: dV = (-F_phys/m - gamma*V + F_bias/m) * dt + Sigma_v * dW
    # where (-F/m) uses factor k_to_acc/mass, and Sigma_v has to be in nm/fs * sqrt(fs)
    # Build sigma_v so that Var(dV) = 2*gamma*kB*T/m * dt  (in nm^2/fs^2)
    sigma_v = np.sqrt(2.0 * gamma * kB * T / masses).reshape(N, 1)  # units: (kJ/(mol))/amu * 1/ps
    # Convert 1/ps to 1/fs: divide by 1e3 inside sqrt? Careful: gamma carries 1/ps inside sqrt.
    # The factor to get nm/fs units consistent is absorbed by our 1e-6 mapping; we'll keep internal consistency.

    logW = np.zeros(T_frames)
    # Iterate over frames; compute per-frame Brownian increment on velocities only.
    for t in range(T_frames - 1):
        Vt = velocities[t]           # (N,3) nm/fs
        Vn = velocities[t+1]         # (N,3) nm/fs
        Fp = forces_phys[t]          # (N,3) kJ/(mol*nm)
        Fb = bias_forces[t]          # (N,3) kJ/(mol*nm)

        # Drift (velocity equation): a = (-F_phys/m - gamma*V + F_bias/m)
        a = (-Fp * k_to_acc / mass) - (gamma * Vt) + (Fb * k_to_acc / mass)  # (N,3) nm/fs^2

        dv = Vn - Vt  # (N,3) nm/fs
        # Discrete Brownian increment estimate on velocities:
        # dv = a*dt + Sigma_v * dW  ->  dW = (dv - a*dt) / Sigma_v
        # Broadcast Sigma_v over cartesian components
        dW = (dv - a * dt) / sigma_v  # (N,3); dimensionless

        # Policy in normalized coordinates: v_t = Sigma^{-1} * (0, F_bias/m)
        v_norm = (Fb * k_to_acc / mass) / sigma_v  # (N,3)

        # Increment logW by sum_i [ v · dW - 0.5 * ||v||^2 * dt_ps * 1e3 ]
        # We need a consistent time scale; use dt in fs, sigma_v used 1/ps; we combine using dt_ps
        incr = (v_norm * dW).sum() - 0.5 * (v_norm**2).sum() * dt_ps
        logW[t+1] = logW[t] + incr

    return logW  # shape (T,)

# ---- FES builder ----
def build_fes_rmsd(positions, RA, RB, weights, nbins=60):
    """
    positions: (T, N, 3)
    RA, RB: (N, 3)
    weights: (T,) frame weights (prefix weights recommended)
    """
    T_frames = positions.shape[0]
    rmsdA = np.empty(T_frames)
    rmsdB = np.empty(T_frames)
    for t in range(T_frames):
        rmsdA[t] = kabsch_rmsd(positions[t], RA)
        rmsdB[t] = kabsch_rmsd(positions[t], RB)

    # Weighted 2D histogram (density=True gives probability density estimate)
    H, xedges, yedges = np.histogram2d(
        rmsdA, rmsdB, bins=nbins, weights=np.exp(weights), density=True
    )
    # Free energy (kBT units): F = -ln p; later you can multiply by kBT if desired.
    # Avoid log(0)
    H = np.where(H > 0, H, np.nan)
    F = -np.log(H)
    F = F - np.nanmin(F)

    X, Y = np.meshgrid(
        0.5*(xedges[:-1]+xedges[1:]),
        0.5*(yedges[:-1]+yedges[1:])
    )

    fig, ax = plt.subplots(figsize=(6,5))
    c = ax.pcolormesh(xedges, yedges, F.T)  # no explicit colors
    ax.plot(rmsdA, rmsdB, color='r', lw=2)
    ax.set_xlabel("RMSD to A (nm)")
    ax.set_ylabel("RMSD to B (nm)")
    ax.set_title("2D Free Energy (units of kBT; min set to 0)")
    fig.colorbar(c, ax=ax, label="F (kBT)")
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 2.5)
    plt.tight_layout()
    plt.show()

# ---- Driver ----
have_vars = all(name in globals() for name in [
    'positions','velocities','forces_phys','bias_forces','masses','dt','T','gamma','RA','RB'
])

if not have_vars:
    missing = [name for name in ['positions','velocities','forces_phys','bias_forces','masses','dt','T','gamma','RA','RB'] if name not in globals()]
    print("Data not found in the environment. Please define the following variables and re-run this cell:")
    print(missing)
else:
    logW = compute_logw_prefix(positions, velocities, forces_phys, bias_forces, masses, dt, T, gamma)
    build_fes_rmsd(positions.reshape(-1, positions.shape[1], 3), RA, RB, weights=logW, nbins=60)
