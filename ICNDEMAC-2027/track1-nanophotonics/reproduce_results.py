"""Harvest real Track-1 (nano-photonics) results for the ICNDEMAC paper.

Direct-gap LED modelling with the FEM drift-diffusion + radiative-recombination
solver: GaAs homojunction, wavelength tuning across III-V materials, and a
GaAs/AlGaAs double-heterostructure quantum LED. Writes figures + results.json.
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DDM_SPC import DriftDiffusionSolver, MaterialLibrary, RectangleMeshBuilder

import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.bbox": "tight",
})
def savefig(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=200)
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)

results = {}

# ============================================================================
# 1. GaAs homojunction LED: I-V, radiative current, IQE, wavelength
# ============================================================================
def build_homojunction(matname, NA=1e16, ND=1e16, depth=1e-6, LX=4e-6, LY=1e-6):
    region_of = lambda xc, yc: "p-region" if xc < LX/2 else "n-region"
    mesh = RectangleMeshBuilder(LX, LY, nx=48, ny=12).build(
        region_of=region_of, region_names=["p-region", "n-region"])
    mat = MaterialLibrary().load(matname)
    solver = DriftDiffusionSolver(
        mesh=mesh, material=mat,
        doping={"p-region": -NA*1e6, "n-region": ND*1e6},
        contacts={
            "anode":   {"type": "ohmic", "region": "p-region", "voltage": 0.0,
                        "nodes": mesh.boundary_nodes["left"]},
            "cathode": {"type": "ohmic", "region": "n-region", "voltage": 0.0,
                        "nodes": mesh.boundary_nodes["right"]},
        },
        recombination="srh+auger+radiative", device_depth=depth)
    return solver, mat

print("=== GaAs homojunction LED ===")
solver, mat = build_homojunction("gaas")
VT = mat.thermal_voltage
sol = solver.solve_equilibrium()
Vbi_num = float(sol.potential.max()-sol.potential.min())
Vbi_ana = float(VT*np.log((1e16*1e6)*(1e16*1e6)/mat.ni**2))
Vgrid = np.round(np.arange(0.05, 1.351, 0.05), 3)
rec = []
t0 = time.perf_counter()
for V in Vgrid:
    solver.solve_bias({"anode": float(V)})
    I = solver.terminal_current("anode")
    opt = solver.optical()
    rec.append((V, I, opt.radiative_current, opt.iqe, opt.optical_power, opt.wavelength_nm))
tsweep = time.perf_counter()-t0
rec = np.array(rec, dtype=float)
V, I, Irad, IQE, Popt, lam = rec.T
opt = solver.optical()
results["gaas_homojunction"] = {
    "Eg_eV": mat.band_gap, "ni_cm3": mat.ni/1e6, "B_m3_s": float(mat.B),
    "Vbi_numeric_V": round(Vbi_num, 4), "Vbi_analytic_V": round(Vbi_ana, 4),
    "wavelength_nm": round(float(opt.wavelength_nm), 1),
    "photon_energy_eV": round(float(opt.photon_energy_eV), 3),
    "peak_IQE": round(float(IQE.max()), 4),
    "I_at_1.3V_A": float(np.interp(1.3, V, I)),
    "Irad_at_1.3V_A": float(np.interp(1.3, V, Irad)),
    "Popt_at_1.3V_W": float(np.interp(1.3, V, Popt)),
    "sweep_time_s": round(tsweep, 2),
    "nodes": int(solver.mesh.num_nodes),
}
print("  Vbi num %.3f vs ana %.3f V; lambda=%.1f nm; peak IQE=%.4f"
      % (Vbi_num, Vbi_ana, opt.wavelength_nm, IQE.max()))

# --- Figure 1: GaAs LED I-V total vs radiative ------------------------------
fwd = I > 0
fig, ax = plt.subplots(figsize=(5.4, 4.0))
ax.semilogy(V[fwd], I[fwd], "-o", ms=4, color="#1f77b4", label="Total current $I$")
ax.semilogy(V[fwd], Irad[fwd], "-s", ms=4, color="#d62728",
            label=r"Radiative $I_{\mathrm{rad}}=qR_{\mathrm{rad}}$")
ax.set_xlabel("Forward bias $V$ (V)")
ax.set_ylabel("Current (A)")
ax.legend(frameon=False)
ax.set_title("GaAs LED: total vs radiative current")
savefig(fig, "fig_gaas_iv")

# --- Figure 2: IQE vs drive current -----------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 4.0))
m = I > 0
ax.semilogx(I[m], IQE[m], "-o", ms=4, color="#2ca02c")
ax.set_xlabel("Drive current $I$ (A)")
ax.set_ylabel("Internal quantum efficiency")
ax.set_ylim(0, 1.02)
ax.set_title("GaAs LED efficiency vs injection")
savefig(fig, "fig_gaas_iqe")

# ============================================================================
# 2. Wavelength tuning across direct-gap materials
# ============================================================================
print("\n=== wavelength tuning across III-V direct-gap materials ===")
tune = []
for mname in ["gaas", "algaas", "gan"]:
    try:
        s, m = build_homojunction(mname, NA=1e16, ND=1e16)
        s.solve_equilibrium()
        # ramp to a forward bias above the (material-dependent) built-in potential
        Vbi = m.thermal_voltage*np.log((1e22)*(1e22)/m.ni**2)
        Vtarget = min(Vbi*0.95, m.band_gap*0.98)
        for Vv in np.linspace(0.1, Vtarget, 14):
            s.solve_bias({"anode": float(Vv)})
        o = s.optical()
        tune.append((mname, m.band_gap, float(o.wavelength_nm), float(o.iqe)))
        print(f"  {mname:7s} Eg={m.band_gap:.2f} eV -> {o.wavelength_nm:.0f} nm (IQE {o.iqe:.3f})")
    except Exception as e:
        print(f"  {mname:7s} FAILED: {type(e).__name__}: {e}")
results["wavelength_tuning"] = [
    {"material": t[0], "Eg_eV": t[1], "wavelength_nm": round(t[2], 1), "IQE": round(t[3], 4)}
    for t in tune]

if tune:
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    names = [t[0].upper() for t in tune]
    waves = [t[2] for t in tune]
    egs = [t[1] for t in tune]
    colors = {"GAAS": "#8c1d1d", "ALGAAS": "#c0392b", "GAN": "#6a0dad"}
    bars = ax.bar(names, waves, color=[colors.get(n, "#555") for n in names], width=0.55)
    for b, eg, w in zip(bars, egs, waves):
        ax.text(b.get_x()+b.get_width()/2, w+15, f"{eg:.2f} eV\n{w:.0f} nm",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Emission wavelength $\\lambda$ (nm)")
    ax.set_ylim(0, max(waves)*1.28)
    ax.set_title("Material-tuned emission (direct-gap III–V)")
    savefig(fig, "fig_wavelength_tuning")

# ============================================================================
# 3. GaAs/AlGaAs double-heterostructure quantum LED (flagship)
# ============================================================================
print("\n=== GaAs/AlGaAs double-heterostructure quantum LED ===")
LX, LY = 300e-9, 10e-9
X1, X2 = 100e-9, 200e-9
def region_of(xc, yc):
    if xc < X1: return "p_clad"
    if xc < X2: return "active"
    return "n_clad"
mesh = RectangleMeshBuilder(LX, LY, nx=60, ny=16).build(
    region_of=region_of, region_names=["p_clad", "active", "n_clad"])
dh = DriftDiffusionSolver(
    mesh=mesh,
    region_properties={
        "active": {"material": "gaas",   "doping":  1e22},
        "p_clad": {"material": "algaas", "doping": -1e24},
        "n_clad": {"material": "algaas", "doping":  1e24},
    },
    contacts={
        "anode":   {"type": "ohmic", "region": "p_clad", "voltage": 0.0,
                    "nodes": mesh.boundary_nodes["left"]},
        "cathode": {"type": "ohmic", "region": "n_clad", "voltage": 0.0,
                    "nodes": mesh.boundary_nodes["right"]},
    },
    recombination="srh+auger+radiative",
    quantum={"model": "schrodinger", "confinement": "y", "num_states": 4},
    statistics="fermi-dirac", device_depth=1e-6)
sol = dh.solve_equilibrium()
Vbi_dh = float(sol.potential.max()-sol.potential.min())
Vgrid = np.round(np.arange(0.1, 1.61, 0.1), 3)
recdh = []
t0 = time.perf_counter()
for V in Vgrid:
    dh.solve_bias({"anode": float(V)})
    dh.pack_solution(float(V))
    I = dh.terminal_current("anode")
    o = dh.optical()
    recdh.append((V, I, o.radiative_current, o.iqe, o.wavelength_nm))
tdh = time.perf_counter()-t0
recdh = np.array(recdh, dtype=float)
Vd, Id, Iradd, IQEd, lamd = recdh.T
odh = dh.optical()
En = np.asarray(dh.quantum.energies_n)[:4]
results["dh_quantum_led"] = {
    "Vbi_V": round(Vbi_dh, 4),
    "subband_energies_eV": [float(round(x, 4)) for x in En],
    "wavelength_nm": round(float(odh.wavelength_nm), 1),
    "photon_energy_eV": round(float(odh.photon_energy_eV), 3),
    "peak_IQE": round(float(IQEd.max()), 4),
    "Lambda_min": float(round(dh._gamma_n.min(), 3)),
    "Lambda_max": float(round(dh._gamma_n.max(), 3)),
    "sweep_time_s": round(tdh, 2),
    "nodes": int(dh.mesh.num_nodes), "elements": int(dh.mesh.num_elements),
}
print("  Vbi=%.3f V; lambda=%.1f nm; peak IQE=%.4f (GaAs active region)"
      % (Vbi_dh, odh.wavelength_nm, IQEd.max()))

# --- Figure 4: carrier confinement in the active region ---------------------
# horizontal slice at mid-body: n(x) piles up in the GaAs active region
store = dh.solution["bias_voltage"]
kb = max(store)
nfield = np.asarray(store[kb]["electron_concentration"])
pfield = np.asarray(store[kb]["hole_concentration"])
nodes = dh.physical_mesh.nodes
ymid = nodes[np.argmin(np.abs(nodes[:,1]-LY/2)),1]
row = np.isclose(nodes[:,1], ymid, atol=1e-12)
order = np.argsort(nodes[row,0])
xs = nodes[row,0][order]*1e9
nx_ = nfield[row][order]; px_ = pfield[row][order]
fig, ax = plt.subplots(figsize=(5.4, 4.0))
ax.semilogy(xs, nx_/1e6, "-", color="#d62728", lw=2, label="electrons $n$")
ax.semilogy(xs, px_/1e6, "-", color="#1f77b4", lw=2, label="holes $p$")
ax.axvspan(X1*1e9, X2*1e9, color="#f1c40f", alpha=0.25, label="GaAs active")
ax.set_xlabel("Transport position $x$ (nm)")
ax.set_ylabel("Carrier density (cm$^{-3}$)")
ax.legend(frameon=False, fontsize=9, loc="lower center")
ax.set_title("DH carrier confinement in the active region")
savefig(fig, "fig_dh_confinement")

# --- Figure 5: DH LED IQE vs current, radiative fraction --------------------
fig, ax = plt.subplots(figsize=(5.4, 4.0))
mm = Id > 0
ax.semilogx(Id[mm], IQEd[mm], "-o", ms=4, color="#2ca02c")
ax.set_xlabel("Drive current $I$ (A)")
ax.set_ylabel("Internal quantum efficiency")
ax.set_ylim(0, 1.02)
ax.set_title("DH LED efficiency vs injection")
savefig(fig, "fig_dh_iqe")

with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n=== RESULTS ===")
print(json.dumps(results, indent=2))
print("\nWritten to", OUT)
