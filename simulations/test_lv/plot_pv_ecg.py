import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Load data
pv = np.loadtxt('output_lv_ellipsoid/pv_loop.csv', delimiter=',', skiprows=1)
t_ms = pv[:, 0]
lvp  = pv[:, 1]
lvv  = pv[:, 2]

ecg = np.loadtxt('output_lv_ellipsoid/pseudo_ecg.csv', delimiter=',', skiprows=1)
t_ecg = ecg[:, 0]
with open('output_lv_ellipsoid/pseudo_ecg.csv') as f:
    lead_names = f.readline().strip().split(',')[1:]

lead_data = {name: ecg[:, i+1] for i, name in enumerate(lead_names)}

# Layout: [PV | P/V traces | ECG grid 4 cols x 3 rows]
fig = plt.figure(figsize=(20, 8))
gs = gridspec.GridSpec(3, 6, figure=fig, wspace=0.4, hspace=0.5)

# ── PV loop ────────────────────────────────────────────────────────────────────
ax_pv = fig.add_subplot(gs[:, 0])
ax_pv.plot(lvv, lvp, 'b-', linewidth=1.5)
ax_pv.set_xlabel('LV Volume (mm³)')
ax_pv.set_ylabel('LV Pressure (kPa)')
ax_pv.set_title('PV Loop')
ax_pv.grid(True, alpha=0.3)

# ── Pressure and volume traces ─────────────────────────────────────────────────
ax_p = fig.add_subplot(gs[0:2, 1])
ax_p.plot(t_ms, lvp, 'b-', linewidth=1.0)
ax_p.set_ylabel('Pressure (kPa)')
ax_p.set_title('LV Pressure')
ax_p.grid(True, alpha=0.3)
ax_p.set_xticklabels([])

ax_v = fig.add_subplot(gs[2, 1])
ax_v.plot(t_ms, lvv, 'r-', linewidth=1.0)
ax_v.set_ylabel('Volume (mm³)')
ax_v.set_xlabel('Time (ms)')
ax_v.set_title('LV Volume')
ax_v.grid(True, alpha=0.3)

# ── ECG grid ───────────────────────────────────────────────────────────────────
ecg_layout = [
    ['I',   'aVR', 'V1', 'V4'],
    ['II',  'aVL', 'V2', 'V5'],
    ['III', 'aVF', 'V3', 'V6'],
]

for row, leads_row in enumerate(ecg_layout):
    for col, lead in enumerate(leads_row):
        ax = fig.add_subplot(gs[row, col + 2])
        if lead in lead_data:
            ax.plot(t_ecg, lead_data[lead], 'k-', linewidth=0.8)
        ax.set_title(lead, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if row < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('ms', fontsize=7)
        if col > 0:
            ax.set_yticklabels([])

plt.suptitle('LV Ellipsoid — Cardiac Cycle Summary', fontsize=12, fontweight='bold')
plt.savefig('output_lv_ellipsoid/results_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved results_summary.png')