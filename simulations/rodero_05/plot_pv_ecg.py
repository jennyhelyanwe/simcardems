import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import os

PV_FILE  = 'results_4mm/default/biv_coarse_run_output/pv_loop.csv'
ECG_FILE = 'results_4mm/default/biv_coarse_run_output/pseudo_ecg.csv'
POLL_INTERVAL = 2.0

PHASE_NAMES = {0: 'Preload', 1: 'IVC', 2: 'Ejection', 3: 'IVR', 4: 'Filling'}

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
fig = plt.figure(figsize=(14, 8))
gs  = gridspec.GridSpec(3, 7, figure=fig, wspace=0.45, hspace=0.55)

# LV PV loop
ax_lv_pv = fig.add_subplot(gs[:, 0])
ax_lv_pv.set_xlabel('LV Volume (mL)')
ax_lv_pv.set_ylabel('LV Pressure (kPa)')
ax_lv_pv.set_title('LV PV Loop')
ax_lv_pv.grid(True, alpha=0.3)

# RV PV loop
ax_rv_pv = fig.add_subplot(gs[:, 1])
ax_rv_pv.set_xlabel('RV Volume (mL)')
ax_rv_pv.set_ylabel('RV Pressure (kPa)')
ax_rv_pv.set_title('RV PV Loop')
ax_rv_pv.grid(True, alpha=0.3)

# Pressure traces
ax_p = fig.add_subplot(gs[0, 2])
ax_p.set_ylabel('Pressure (kPa)')
ax_p.set_title('LV & RV Pressure')
ax_p.grid(True, alpha=0.3)

# Volume traces
ax_v = fig.add_subplot(gs[1, 2])
ax_v.set_ylabel('Volume (mm³)')
ax_v.set_title('LV & RV Volume')
ax_v.grid(True, alpha=0.3)

# Phase monitor
ax_phase = fig.add_subplot(gs[2, 2])
ax_phase.set_ylabel('Phase')
ax_phase.set_xlabel('Time (ms)')
ax_phase.set_title('Phase')
ax_phase.set_yticks([0, 1, 2, 3, 4])
ax_phase.set_yticklabels(['Preload', 'IVC', 'Eject', 'IVR', 'Fill'], fontsize=7)
ax_phase.grid(True, alpha=0.3)

ecg_layout = [
    ['I',   'aVR', 'V1', 'V4'],
    ['II',  'aVL', 'V2', 'V5'],
    ['III', 'aVF', 'V3', 'V6'],
]
ecg_axes = {}
for row, leads_row in enumerate(ecg_layout):
    for col, lead in enumerate(leads_row):
        ax = fig.add_subplot(gs[row, col + 3])
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

plt.suptitle('BiV — Live Cardiac Cycle Monitor', fontsize=12, fontweight='bold')

while True:
    pv   = load_csv(PV_FILE)
    ecg  = load_csv(ECG_FILE)
    lead_names = get_lead_names(ECG_FILE)

    if pv is not None and len(pv.shape) == 2 and pv.shape[0] > 1:
        t_ms = pv[:, 0]
        lvp  = pv[:, 1]
        lvv  = pv[:, 2]/1000
        rvp  = pv[:, 3]
        rvv  = pv[:, 4]/1000
        #lv_phase = pv[:, 5]
        #rv_phase = pv[:, 6]

        # LV PV loop
        ax_lv_pv.cla()
        ax_lv_pv.plot(lvv, lvp, 'b-', linewidth=1.5)
        ax_lv_pv.set_xlabel('LV Volume (mm³)')
        ax_lv_pv.set_ylabel('LV Pressure (kPa)')
        ax_lv_pv.set_title('LV PV Loop')
        ax_lv_pv.grid(True, alpha=0.3)

        # RV PV loop
        ax_rv_pv.cla()
        ax_rv_pv.plot(rvv, rvp, 'g-', linewidth=1.5)
        ax_rv_pv.set_xlabel('RV Volume (mm³)')
        ax_rv_pv.set_ylabel('RV Pressure (kPa)')
        ax_rv_pv.set_title('RV PV Loop')
        ax_rv_pv.grid(True, alpha=0.3)

        # Pressure traces
        ax_p.cla()
        ax_p.plot(t_ms, lvp, 'b-', linewidth=1.0, label='LV')
        ax_p.plot(t_ms, rvp, 'g-', linewidth=1.0, label='RV')
        ax_p.set_ylabel('Pressure (kPa)')
        ax_p.set_title('LV & RV Pressure')
        ax_p.legend(fontsize=7)
        ax_p.grid(True, alpha=0.3)

        # Volume traces
        ax_v.cla()
        ax_v.plot(t_ms, lvv, 'b-', linewidth=1.0, label='LV')
        ax_v.plot(t_ms, rvv, 'g-', linewidth=1.0, label='RV')
        ax_v.set_ylabel('Volume (mm³)')
        ax_v.set_title('LV & RV Volume')
        ax_v.legend(fontsize=7)
        ax_v.grid(True, alpha=0.3)

        # Phase traces
        ax_phase.cla()
        #ax_phase.step(t_ms, lv_phase, 'b-', where='post', linewidth=1.2, label='LV')
        #ax_phase.step(t_ms, rv_phase, 'g-', where='post', linewidth=1.2, label='RV')
        ax_phase.set_ylabel('Phase')
        ax_phase.set_xlabel('Time (ms)')
        ax_phase.set_title('Phase')
        ax_phase.set_yticks([0, 1, 2, 3, 4])
        ax_phase.set_yticklabels(['Preload', 'IVC', 'Eject', 'IVR', 'Fill'], fontsize=7)
        ax_phase.legend(fontsize=7)
        ax_phase.grid(True, alpha=0.3)

    if ecg is not None and len(lead_names) > 0 and len(ecg.shape) == 2 and ecg.shape[0] > 1:
        t_ecg = ecg[:, 0]
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
        plt.ioff()
        plt.savefig('biv_coarse_run_output/results_summary.png', dpi=150, bbox_inches='tight')
        plt.show()
        break
