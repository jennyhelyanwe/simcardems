import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import os

PV_FILE  = 'output_lv_ellipsoid_isovol_lagrange/pv_loop.csv'
ECG_FILE = 'output_lv_ellipsoid_isovol_lagrange/pseudo_ecg.csv'
POLL_INTERVAL = 2.0  # seconds between updates

def load_csv(path):
    try:
        return np.loadtxt(path, delimiter=',', skiprows=1)
    except Exception:
        return None

def get_lead_names(path):
    try:
        with open(path) as f:
            return f.readline().strip().split(',')[1:]
    except Exception:
        return []

plt.ion()
fig = plt.figure(figsize=(20, 8))
gs  = gridspec.GridSpec(3, 6, figure=fig, wspace=0.4, hspace=0.5)

ax_pv = fig.add_subplot(gs[:, 0])
ax_p  = fig.add_subplot(gs[0:2, 1])
ax_v  = fig.add_subplot(gs[2, 1])

ecg_layout = [
    ['I',   'aVR', 'V1', 'V4'],
    ['II',  'aVL', 'V2', 'V5'],
    ['III', 'aVF', 'V3', 'V6'],
]
ecg_axes = {}
for row, leads_row in enumerate(ecg_layout):
    for col, lead in enumerate(leads_row):
        ax = fig.add_subplot(gs[row, col + 2])
        ax.set_title(lead, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if row < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('ms', fontsize=7)
        if col > 0:
            ax.set_yticklabels([])
        ecg_axes[lead] = ax

plt.suptitle('LV Ellipsoid — Live Cardiac Cycle Monitor', fontsize=12, fontweight='bold')

while True:
    pv   = load_csv(PV_FILE)
    ecg  = load_csv(ECG_FILE)
    lead_names = get_lead_names(ECG_FILE)

    if pv is not None and len(pv.shape) == 2 and pv.shape[0] > 1:
        t_ms = pv[:, 0]
        lvp  = pv[:, 1]
        lvv  = pv[:, 2]

        ax_pv.cla()
        ax_pv.plot(lvv, lvp, 'b-', linewidth=1.5)
        ax_pv.set_xlabel('LV Volume (mm³)')
        ax_pv.set_ylabel('LV Pressure (kPa)')
        ax_pv.set_title('PV Loop')
        ax_pv.grid(True, alpha=0.3)

        ax_p.cla()
        ax_p.plot(t_ms, lvp, 'b-', linewidth=1.0)
        ax_p.set_ylabel('Pressure (kPa)')
        ax_p.set_title('LV Pressure')
        ax_p.grid(True, alpha=0.3)

        ax_v.cla()
        ax_v.plot(t_ms, lvv, 'r-', linewidth=1.0)
        ax_v.set_ylabel('Volume (mm³)')
        ax_v.set_xlabel('Time (ms)')
        ax_v.set_title('LV Volume')
        ax_v.grid(True, alpha=0.3)

    if ecg is not None and len(lead_names) > 0 and len(ecg.shape) == 2 and ecg.shape[0] > 1:
        t_ecg    = ecg[:, 0]
        lead_data = {name: ecg[:, i+1] for i, name in enumerate(lead_names)}

        for lead, ax in ecg_axes.items():
            ax.cla()
            ax.set_title(lead, fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if lead in lead_data:
                ax.plot(t_ecg, lead_data[lead], 'k-', linewidth=0.8)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(POLL_INTERVAL)

    if pv is not None and len(pv.shape) == 2 and pv[-1, 0] >= 800.0:
        mpi_print("Simulation complete.")
        plt.ioff()
        plt.savefig('output_lv_ellipsoid/results_summary.png', dpi=150, bbox_inches='tight')
        plt.show()
        break
