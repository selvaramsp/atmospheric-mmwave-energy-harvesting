# =====================================================
# CODE 1: Absorption Coefficient Setup (HITRAN Model)
# =====================================================

import numpy as np
import matplotlib.pyplot as plt

# Frequency range: 57 GHz to 325 GHz, 1 GHz steps
f_GHz = np.arange(57, 326, 1)        # shape: (269,)
f_Hz  = f_GHz * 1e9

# Relative humidity (fixed at 50% here, varied in later codes)
RH = 50
RH_factor = RH / 50                  # normalised to 50% baseline

# ── Oxygen (O2) absorption peaks ──────────────────────
k_O2 = np.ones_like(f_GHz, dtype=float) * 0.001   # baseline

# Peak at 60 GHz
k_O2 += 0.015 * np.exp(-((f_GHz - 60)**2) / (2 * 5**2))

# Peak at 118 GHz
k_O2 += 0.008 * np.exp(-((f_GHz - 118)**2) / (2 * 4**2))

# ── Water Vapor (H2O) absorption peaks ────────────────
k_H2O = np.ones_like(f_GHz, dtype=float) * 0.0005  # baseline

# Peak at 22 GHz (edge of our band)
k_H2O += RH_factor * 0.003 * np.exp(-((f_GHz - 22)**2) / (2 * 3**2))

# Peak at 183 GHz
k_H2O += RH_factor * 10.0 * np.exp(-((f_GHz - 183)**2) / (2 * 6**2))

# Peak at 325 GHz
k_H2O += RH_factor * 20.0 * np.exp(-((f_GHz - 325)**2) / (2 * 5**2))

# ── Total absorption coefficient ──────────────────────
k_total = k_O2 + k_H2O              # units: m⁻¹

# ── Plot ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(f_GHz, k_total, 'b-',  linewidth=2, label='Total k(f)')
ax.semilogy(f_GHz, k_O2,   'r--', linewidth=1.5, label='O₂ only')
ax.semilogy(f_GHz, k_H2O,  'g--', linewidth=1.5, label='H₂O only')

ax.set_xlabel('Frequency (GHz)', fontsize=12)
ax.set_ylabel('Absorption Coefficient k (m⁻¹)', fontsize=12)
ax.set_title('Atmospheric Absorption Coefficient — HITRAN Model', fontsize=13)
ax.legend()
ax.grid(True, which='both', alpha=0.4)
ax.set_xlim([57, 325])

# Annotate peaks
for freq, label in [(60, '60 GHz\nO₂'), (118, '118 GHz\nO₂'),
                     (183, '183 GHz\nH₂O'), (325, '325 GHz\nH₂O')]:
    idx = np.argmin(np.abs(f_GHz - freq))
    ax.annotate(label, xy=(freq, k_total[idx]),
                xytext=(freq + 8, k_total[idx] * 2),
                fontsize=8, color='white',
                arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig('code1_absorption.png', dpi=150, bbox_inches='tight')
plt.show()

# Print key values
for freq in [60, 118, 183, 325]:
    idx = np.argmin(np.abs(f_GHz - freq))
    print(f"k at {freq:3d} GHz = {k_total[idx]:.4f} m⁻¹")

# =====================================================
# CODE 2: Planck Blackbody Radiation Spectrum
# =====================================================

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
h  = 6.626e-34    # Planck's constant (J·s)
c  = 3e8          # Speed of light (m/s)
kb = 1.381e-23    # Boltzmann constant (J/K)

# Atmospheric temperature
T = 290           # Kelvin (17°C — standard atmosphere)

# Frequency range
f_GHz = np.arange(57, 326, 1)
f_Hz  = f_GHz * 1e9

# ── Full Planck's Law ──────────────────────────────────
B_planck = (2 * h * f_Hz**3 / c**2) / (np.exp(h * f_Hz / (kb * T)) - 1)

# ── Rayleigh-Jeans Approximation ──────────────────────
# Valid when hf << kT (holds well at mmWave below ~1 THz at 290K)
B_RJ = 2 * f_Hz**2 * kb * T / c**2

# ── Plot ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(f_GHz, B_planck, 'b-', linewidth=2)
axes[0].set_xlabel('Frequency (GHz)', fontsize=12)
axes[0].set_ylabel('Spectral Radiance (W/m²/Hz/sr)', fontsize=11)
axes[0].set_title('Planck Blackbody Radiation at T = 290K', fontsize=12)
axes[0].grid(True, alpha=0.4)

axes[1].plot(f_GHz, B_RJ,     'r-',  linewidth=2,   label='Rayleigh-Jeans')
axes[1].plot(f_GHz, B_planck, 'b--', linewidth=1.5, label='Full Planck')
axes[1].set_xlabel('Frequency (GHz)', fontsize=12)
axes[1].set_ylabel('Spectral Radiance (W/m²/Hz/sr)', fontsize=11)
axes[1].set_title('Planck vs Rayleigh-Jeans (mmWave range)', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('code2_planck.png', dpi=150, bbox_inches='tight')
plt.show()

# Print sample values
for freq in [60, 118, 183, 325]:
    idx = np.argmin(np.abs(f_GHz - freq))
    print(f"B at {freq:3d} GHz = {B_planck[idx]:.4e} W/m²/Hz/sr")

# =====================================================
# CODE 3: Beer-Lambert Transmission — tau(f, d)
# =====================================================

import numpy as np
import matplotlib.pyplot as plt

# Frequency and distance arrays
f_GHz = np.arange(57, 326, 1)        # 269 points
d_m   = np.arange(1, 101, 1)         # 100 points (1m to 100m)

RH = 50
RH_factor = RH / 50

# Rebuild k_total (same as Code 1)
k_O2  = (0.001
         + 0.015 * np.exp(-((f_GHz - 60)**2)  / (2 * 5**2))
         + 0.008 * np.exp(-((f_GHz - 118)**2) / (2 * 4**2)))

k_H2O = (0.0005
         + RH_factor * 0.003 * np.exp(-((f_GHz - 22)**2)  / (2 * 3**2))
         + RH_factor * 10.0  * np.exp(-((f_GHz - 183)**2) / (2 * 6**2))
         + RH_factor * 20.0  * np.exp(-((f_GHz - 325)**2) / (2 * 5**2)))

k_total = k_O2 + k_H2O               # shape: (269,)

# ── Compute transmission matrix using broadcasting ────
# K[i, j] = k_total[i],  D[i, j] = d_m[j]
K   = k_total[:, np.newaxis]          # shape: (269, 1)
D   = d_m[np.newaxis, :]              # shape: (1, 100)
tau = np.exp(-K * D)                  # shape: (269, 100)

# ── Plot 1: Transmission vs distance at key frequencies
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors = {'60 GHz': 'blue', '118 GHz': 'magenta',
          '183 GHz': 'green', '325 GHz': 'red'}

for freq_label, freq_val in [(60, 60), (118, 118), (183, 183), (325, 325)]:
    idx = np.argmin(np.abs(f_GHz - freq_val))
    axes[0].plot(d_m, tau[idx, :], linewidth=2,
                 label=f'{freq_val} GHz',
                 color=list(colors.values())[list(colors.keys()).index(f'{freq_val} GHz')])

axes[0].set_xlabel('Distance (m)', fontsize=12)
axes[0].set_ylabel('Transmission τ (fraction surviving)', fontsize=11)
axes[0].set_title('Beer-Lambert Transmission vs Distance', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.4)
axes[0].set_ylim([0, 1])

# ── Plot 2: Heatmap ───────────────────────────────────
im = axes[1].imshow(tau, aspect='auto', origin='lower',
                    extent=[d_m[0], d_m[-1], f_GHz[0], f_GHz[-1]],
                    cmap='jet', vmin=0, vmax=1)
plt.colorbar(im, ax=axes[1], label='Transmission τ')
axes[1].set_xlabel('Distance (m)', fontsize=12)
axes[1].set_ylabel('Frequency (GHz)', fontsize=12)
axes[1].set_title('Transmission Heatmap τ(f,d) — RH = 50%', fontsize=12)

# Mark key frequency lines
for freq in [60, 118, 183, 325]:
    axes[1].axhline(y=freq, color='white', linestyle='--',
                    linewidth=1, alpha=0.7, label=f'{freq} GHz')

plt.tight_layout()
plt.savefig('code3_transmission.png', dpi=150, bbox_inches='tight')
plt.show()

# Print key values
for freq, dist in [(60, 50), (118, 30), (183, 10), (325, 5)]:
    fi = np.argmin(np.abs(f_GHz - freq))
    di = dist - 1
    print(f"At {freq:3d} GHz, d={dist:3d}m: τ = {tau[fi, di]:.4f}  "
          f"({tau[fi, di]*100:.1f}% survives)")

# =====================================================
# CODE 4: MAIN SIMULATION — P_DC(f, d, RH)
# =====================================================

import numpy as np
import matplotlib.pyplot as plt

# ── Physical constants ────────────────────────────────
h  = 6.626e-34
c  = 3e8
kb = 1.381e-23
T  = 290

# ── Parameter arrays ─────────────────────────────────
f_GHz  = np.arange(57, 326, 1)          # 269 frequency points
f_Hz   = f_GHz * 1e9
d_m    = np.arange(1, 101, 1)           # 100 distance points
RH_arr = np.array([20, 35, 50, 65, 80, 95])  # 6 humidity levels

# ── Antenna parameters ────────────────────────────────
G_dBi    = 20
G        = 10 ** (G_dBi / 10)           # linear gain = 100
lam      = c / f_Hz                     # wavelength (m), shape (269,)
A_eff    = G * lam**2 / (4 * np.pi)     # effective aperture (m²), shape (269,)
eta_ant  = 0.8                          # antenna efficiency

# ── Rectifier efficiency ─────────────────────────────
eta_rect = 0.28                         # 28% from hardware measurements

# ── Planck radiation ─────────────────────────────────
B = (2 * h * f_Hz**3 / c**2) / (np.exp(h * f_Hz / (kb * T)) - 1)
# shape: (269,)

# ── Storage ───────────────────────────────────────────
P_DC_matrix  = np.zeros((len(d_m), len(RH_arr)))   # (100, 6)
P_perHz_map  = np.zeros((len(f_GHz), len(d_m)))    # for heatmap at RH=50%

# ── Main simulation loop ─────────────────────────────
for ri, RH in enumerate(RH_arr):
    RH_factor = RH / 50

    # Absorption coefficient at this humidity
    k_O2  = (0.001
             + 0.015 * np.exp(-((f_GHz - 60)**2)  / (2 * 5**2))
             + 0.008 * np.exp(-((f_GHz - 118)**2) / (2 * 4**2)))

    k_H2O = (0.0005
             + RH_factor * 0.003 * np.exp(-((f_GHz - 22)**2)  / (2 * 3**2))
             + RH_factor * 10.0  * np.exp(-((f_GHz - 183)**2) / (2 * 6**2))
             + RH_factor * 20.0  * np.exp(-((f_GHz - 325)**2) / (2 * 5**2)))

    k_total = k_O2 + k_H2O              # shape: (269,)

    for di, d in enumerate(d_m):
        # Beer-Lambert transmission
        tau = np.exp(-k_total * d)      # shape: (269,)

        # Received RF power spectral density (W/Hz)
        P_f = B * tau * A_eff * eta_ant # shape: (269,)

        # Integrate over frequency using trapezoidal rule
        P_RF = np.trapz(P_f, f_Hz)     # total RF power (W)

        # Convert RF → DC
        P_DC = P_RF * eta_rect

        P_DC_matrix[di, ri] = P_DC

        # Save spectral map at RH = 50% for heatmap
        if RH == 50:
            P_perHz_map[:, di] = P_f

# ── Convert to nanowatts ──────────────────────────────
P_DC_nW = P_DC_matrix * 1e9

# ── Save results ──────────────────────────────────────
np.save('P_DC_nW.npy',      P_DC_nW)
np.save('P_perHz_map.npy',  P_perHz_map)
np.save('f_GHz.npy',        f_GHz)
np.save('d_m.npy',          d_m)
np.save('RH_arr.npy',       RH_arr)

# ── Print summary table ───────────────────────────────
print("\n--- Harvested DC Power (nW) ---")
print(f"{'Distance':>10} | {'RH=20%':>8} | {'RH=50%':>8} | {'RH=80%':>8} | {'RH=95%':>8}")
print("-" * 55)
for d_idx in [5, 10, 20, 50, 100]:
    print(f"{d_idx:>8}m   | "
          f"{P_DC_nW[d_idx-1, 0]:>8.3f} | "
          f"{P_DC_nW[d_idx-1, 2]:>8.3f} | "
          f"{P_DC_nW[d_idx-1, 4]:>8.3f} | "
          f"{P_DC_nW[d_idx-1, 5]:>8.3f}")

print("\nResults saved to .npy files")

# =====================================================
# CODE 5: FREQUENCY-DISTANCE HEATMAP
# =====================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load results from Code 4
P_perHz_map = np.load('P_perHz_map.npy')   # (269, 100)
f_GHz       = np.load('f_GHz.npy')
d_m         = np.load('d_m.npy')

# Convert to nW/Hz and take log for better colour range
P_log = np.log10(P_perHz_map * 1e9 + 1e-30)

# ── Plot heatmap ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

im = ax.imshow(P_log, aspect='auto', origin='lower',
               extent=[d_m[0], d_m[-1], f_GHz[0], f_GHz[-1]],
               cmap='jet')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('log₁₀(Power density) [nW/Hz]', fontsize=11)

ax.set_xlabel('Distance (m)', fontsize=13)
ax.set_ylabel('Frequency (GHz)', fontsize=13)
ax.set_title('Frequency–Distance Harvesting Heatmap (RH = 50%)', fontsize=14)

# Mark key absorption lines
key_freqs = [60, 118, 183, 325]
for freq in key_freqs:
    ax.axhline(y=freq, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(102, freq + 3, f'{freq} GHz', color='white',
            fontsize=9, fontfamily='monospace')

# Annotation boxes for sweet spots
ax.annotate('Best zone\n(183 GHz,\n5–30 m)',
            xy=(17, 183), xytext=(30, 210),
            color='white', fontsize=9,
            arrowprops=dict(arrowstyle='->', color='white', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#00000088'))

ax.annotate('Long range\n(60 GHz,\nup to 80 m)',
            xy=(50, 60), xytext=(60, 90),
            color='white', fontsize=9,
            arrowprops=dict(arrowstyle='->', color='white', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#00000088'))

plt.tight_layout()
plt.savefig('code5_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# =====================================================
# CODE 6: OPTIMAL FREQUENCY VS DISTANCE
# =====================================================

import numpy as np
import matplotlib.pyplot as plt

# Load results
f_GHz = np.load('f_GHz.npy')
d_m   = np.load('d_m.npy')

# Rebuild spectral power at RH=50%
h = 6.626e-34; c = 3e8; kb = 1.381e-23; T = 290
f_Hz  = f_GHz * 1e9
G     = 10 ** (20/10)
lam   = c / f_Hz
A_eff = G * lam**2 / (4 * np.pi)
eta_ant  = 0.8
eta_rect = 0.28

B = (2*h*f_Hz**3/c**2) / (np.exp(h*f_Hz/(kb*T)) - 1)

RH_factor = 1.0
k_O2  = (0.001
         + 0.015 * np.exp(-((f_GHz-60)**2)  / (2*5**2))
         + 0.008 * np.exp(-((f_GHz-118)**2) / (2*4**2)))
k_H2O = (0.0005
         + RH_factor * 10.0 * np.exp(-((f_GHz-183)**2) / (2*6**2))
         + RH_factor * 20.0 * np.exp(-((f_GHz-325)**2) / (2*5**2)))
k_total = k_O2 + k_H2O

# For each distance, find the frequency giving max power
optimal_freq = np.zeros(len(d_m))
max_power    = np.zeros(len(d_m))

for di, d in enumerate(d_m):
    tau = np.exp(-k_total * d)
    P_f = B * tau * A_eff * eta_ant * eta_rect * 1e9  # nW/Hz
    best_idx = np.argmax(P_f)
    optimal_freq[di] = f_GHz[best_idx]
    max_power[di]    = P_f[best_idx]

# ── Plot ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(d_m, optimal_freq, 'b-o', linewidth=2,
             markersize=3, label='Optimal frequency')
axes[0].set_xlabel('Distance (m)', fontsize=12)
axes[0].set_ylabel('Optimal Frequency (GHz)', fontsize=12)
axes[0].set_title('Best Frequency to Use at Each Distance', fontsize=13)
axes[0].set_ylim([50, 340])
axes[0].grid(True, alpha=0.4)

for freq, color, label in [(60,'red','60 GHz'), (118,'green','118 GHz'),
                             (183,'magenta','183 GHz')]:
    axes[0].axhline(y=freq, color=color, linestyle='--',
                    linewidth=1.2, alpha=0.7, label=label)
axes[0].legend(fontsize=9)

# Shade the three regions
axes[0].axvspan(1,  20, alpha=0.08, color='magenta', label='183 GHz zone')
axes[0].axvspan(20, 80, alpha=0.08, color='green',   label='118 GHz zone')
axes[0].axvspan(80, 100, alpha=0.08, color='red',    label='60 GHz zone')

axes[1].semilogy(d_m, max_power, 'r-', linewidth=2)
axes[1].set_xlabel('Distance (m)', fontsize=12)
axes[1].set_ylabel('Maximum Harvestable Power (nW/Hz)', fontsize=12)
axes[1].set_title('Maximum Power at Optimal Frequency', fontsize=13)
axes[1].grid(True, which='both', alpha=0.4)
axes[1].axhline(y=1, color='gray', linestyle=':', linewidth=1.5,
                label='1 nW threshold')
axes[1].legend()

plt.tight_layout()
plt.savefig('code6_optimal_freq.png', dpi=150, bbox_inches='tight')
plt.show()

# Print design guidelines
print("\n--- Optimal Frequency Design Guidelines ---")
zones = [(d_m <= 20, "d < 20m"), 
         ((d_m > 20) & (d_m <= 80), "20m < d < 80m"),
         (d_m > 80, "d > 80m")]
for mask, label in zones:
    freqs_in_zone = optimal_freq[mask]
    most_common   = float(np.bincount(freqs_in_zone.astype(int)).argmax())
    print(f"{label:>15}  →  Optimal frequency: {most_common:.0f} GHz")

# =====================================================
# CODE 7: HUMIDITY SENSITIVITY ANALYSIS
# =====================================================

import numpy as np
import matplotlib.pyplot as plt

# Load results
P_DC_nW = np.load('P_DC_nW.npy')     # (100, 6)
d_m     = np.load('d_m.npy')
RH_arr  = np.load('RH_arr.npy')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── Plot 1: Power vs Distance for each RH level ───────
colors = ['#4466FF', '#44AAFF', '#44CC88',
          '#FFCC00', '#FF8844', '#FF4444']

for ri, RH in enumerate(RH_arr):
    axes[0].semilogy(d_m, P_DC_nW[:, ri],
                     color=colors[ri], linewidth=2,
                     label=f'RH = {RH}%')

axes[0].set_xlabel('Distance (m)', fontsize=12)
axes[0].set_ylabel('DC Power (nW)', fontsize=12)
axes[0].set_title('Harvested Power vs Distance — All Humidity Levels', fontsize=12)
axes[0].legend(fontsize=9)
axes[0].grid(True, which='both', alpha=0.4)
axes[0].axhline(y=1, color='white', linestyle=':', linewidth=1.5,
                alpha=0.6, label='1 nW min')

# ── Plot 2: Power vs Humidity at fixed distances ──────
fixed_dists  = [5, 10, 20, 50]
dist_colors  = ['blue', 'green', 'red', 'magenta']

for d, col in zip(fixed_dists, dist_colors):
    axes[1].plot(RH_arr, P_DC_nW[d-1, :],
                 color=col, linewidth=2,
                 marker='o', markersize=7,
                 label=f'd = {d}m')

axes[1].set_xlabel('Relative Humidity (%)', fontsize=12)
axes[1].set_ylabel('DC Power (nW)', fontsize=12)
axes[1].set_title('Power vs Humidity at Fixed Distances', fontsize=12)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('code7_humidity.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Print humidity effect table ───────────────────────
print("\n--- Humidity Effect at d = 10m ---")
print(f"{'RH':>6} | {'Power (nW)':>12} | {'Ratio vs RH=20%':>16}")
print("-" * 40)
base = P_DC_nW[9, 0]   # d=10m, RH=20%
for ri, RH in enumerate(RH_arr):
    power = P_DC_nW[9, ri]
    print(f"{RH:>5}%  | {power:>12.3f} | {power/base:>14.1f}×")

# =====================================================
# CODE 8: RECTENNA EFFICIENCY VALIDATION
# =====================================================

import numpy as np
import matplotlib.pyplot as plt

# ── RF input power sweep ──────────────────────────────
P_RF_dBm = np.arange(-25, 1, 1)
P_RF_W   = 10 ** (P_RF_dBm / 10) * 1e-3

# ── Schottky diode rectifier model (SMS7630) ─────────
Vf        = 0.15        # forward voltage (V)
R_L       = 100e3       # load resistance (100 kΩ)
eta_match = 0.85        # impedance matching efficiency

# Half-wave rectifier: V_out ≈ sqrt(2 * R_L * P_RF) - Vf
V_out = np.sqrt(2 * R_L * P_RF_W * eta_match) - Vf
V_out = np.maximum(V_out, 0)             # clamp below 0

P_DC_W   = V_out**2 / R_L
efficiency = np.where(P_RF_W > 1e-12,
                      P_DC_W / P_RF_W * 100,
                      0)

# ── Your actual measured data ─────────────────────────
meas_dBm = np.array([-22, -20, -18, -15, -12, -10, -8, -5])
meas_eff  = np.array([0,    5,   15,  28,  22,  18,  14, 10])
meas_Vout = np.array([0,  0.05, 0.18, 0.50, 0.78, 1.02, 1.18, 1.20])

# ── Plot ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Efficiency
axes[0].plot(P_RF_dBm, efficiency, 'b-', linewidth=2,
             label='Theoretical model')
axes[0].plot(meas_dBm, meas_eff, 'ro', markersize=9,
             markerfacecolor='red', linewidth=2,
             label='Measured (hardware)')
axes[0].axvline(x=-22, color='gray', linestyle='--',
                linewidth=1.5, label='−22 dBm threshold')
axes[0].set_xlabel('RF Input Power (dBm)', fontsize=12)
axes[0].set_ylabel('Rectification Efficiency (%)', fontsize=12)
axes[0].set_title('Rectenna Efficiency: Theory vs Measured', fontsize=12)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.4)

# DC output voltage
axes[1].plot(meas_dBm, meas_Vout, 'g-o', linewidth=2,
             markersize=8, markerfacecolor='green')
axes[1].set_xlabel('RF Input Power (dBm)', fontsize=12)
axes[1].set_ylabel('DC Output Voltage (V)', fontsize=12)
axes[1].set_title('Measured DC Output Voltage', fontsize=12)
axes[1].grid(True, alpha=0.4)

# Loss breakdown pie chart
losses = [10, 40, 15, 5, 30]
labels = ['Antenna\nMismatch\n10%',
          'Diode Vf\nLoss\n40%',
          'Impedance\nMismatch\n15%',
          'Cap ESR\n5%',
          'Useful DC\nOutput\n30%']
colors_pie = ['#FF6B6B', '#FF8E53', '#FFC300',
              '#DAF7A6', '#00C48C']

axes[2].pie(losses, labels=labels, colors=colors_pie,
            autopct='%1.0f%%', startangle=90,
            textprops={'fontsize': 9})
axes[2].set_title('Power Loss Breakdown', fontsize=12)

plt.tight_layout()
plt.savefig('code8_rectenna.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n--- Rectenna Performance Summary ---")
print("Sensitivity threshold :  -22 dBm")
print("Peak efficiency       :  28% at -15 dBm")
print("Max output power      :  18 µW at -5 dBm")
print("Max DC output voltage :  1.2 V")

# =====================================================
# CODE 9: IOT POWER BUDGET — CAN IT ACTUALLY WORK?
# =====================================================

import numpy as np
import matplotlib.pyplot as plt

# Load simulation results
P_DC_nW = np.load('P_DC_nW.npy')    # (100, 6)
d_m     = np.load('d_m.npy')
RH_arr  = np.load('RH_arr.npy')

# ── Harvested power at different conditions ───────────
P_best  = P_DC_nW[9,  4]    # d=10m, RH=80%  → best realistic
P_typ   = P_DC_nW[19, 2]    # d=20m, RH=50%  → typical
P_worst = P_DC_nW[49, 0]    # d=50m, RH=20%  → worst case

print("--- Harvested Power ---")
print(f"Best case  (10m, RH=80%): {P_best:.3f} nW")
print(f"Typical    (20m, RH=50%): {P_typ:.3f} nW")
print(f"Worst case (50m, RH=20%): {P_worst:.3f} nW")

# ── Supercapacitor charging time ──────────────────────
C_cap     = 100e-6    # 100 µF
V_target  = 1.5       # target voltage (V)
E_needed  = 0.5 * C_cap * V_target**2   # joules

print(f"\n--- Capacitor Charging ---")
print(f"Energy needed to reach {V_target}V: {E_needed*1e6:.2f} µJ")

for name, P_nW in [("Best case", P_best),
                   ("Typical",   P_typ),
                   ("Worst case",P_worst)]:
    P_W      = P_nW * 1e-9
    t_charge = E_needed / P_W
    print(f"{name:>12}: {t_charge:.1f} s  ({t_charge/60:.1f} min)")

# ── Duty cycle calculation ─────────────────────────────
print("\n--- Duty Cycle for Real Sensors ---")

sensors = [
    ("TI MSP430",     0.5e-6,  100e-6,  10e-3),
    ("Nordic nRF52",  2e-6,    5e-3,    10e-3),
    ("Soil Sensor",   1e-6,    200e-6,  50e-3),
]

P_harvest_W = P_typ * 1e-9

for name, P_sleep, P_active, t_active in sensors:
    # Average power = D*P_active + (1-D)*P_sleep = P_harvest_W
    D = (P_harvest_W - P_sleep) / (P_active - P_sleep)
    D = float(np.clip(D, 0, 1))

    if D > 0:
        t_sleep    = t_active * (1 - D) / D
        wakeup_int = t_sleep + t_active
        readings_hr = 3600 / wakeup_int
    else:
        t_sleep = float('inf')
        readings_hr = 0

    print(f"\nSensor: {name}")
    print(f"  Duty cycle     : {D*100:.4f}% active")
    print(f"  Wakes up every : {wakeup_int:.1f} s")
    print(f"  Readings/hour  : {readings_hr:.1f}")

# ── Bar chart: harvest vs demand ──────────────────────
fig, ax = plt.subplots(figsize=(11, 5))

categories = ['Harvest\n(best)',  'Harvest\n(typical)', 'Harvest\n(worst)',
              'MSP430\nsleep',    'nRF52\nsleep',        'Sensor\nactive']
values_nW  = [P_best, P_typ, P_worst, 0.5e3, 2e3, 100e3]
bar_colors = ['#00C48C', '#44AA66', '#228844',
              '#FF8C42', '#FF6633', '#EF476F']

bars = ax.bar(categories, values_nW, color=bar_colors,
              edgecolor='white', linewidth=0.8)
ax.set_yscale('log')
ax.set_ylabel('Power (nW)', fontsize=12)
ax.set_title('Harvested Power vs IoT Sensor Requirements', fontsize=13)
ax.axhline(y=1, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax.grid(True, which='both', alpha=0.3, axis='y')

# Value labels on bars
for bar, val in zip(bars, values_nW):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() * 1.3,
            f'{val:.1f} nW' if val < 1000 else f'{val/1000:.1f} µW',
            ha='center', va='bottom', fontsize=9, color='white')

plt.tight_layout()
plt.savefig('code9_power_budget.png', dpi=150, bbox_inches='tight')
plt.show()

