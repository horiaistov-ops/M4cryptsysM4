import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import hashlib
import time

# --- CONSTANTS & CONFIGURATION ---
SEED_VAL = str(time.time())
SEED_HASH = hashlib.sha256(SEED_VAL.encode()).hexdigest()[:8]

# M4 Architecture Parameters
EXPANSION_RATE = 0.03
ROTATION_SPEED = 0.05
AXIS_ORBIT_SPEED = 0.02 
A3_DRIFT_SPEED = np.array([0.05, 0.02, 0.0]) 
AXIS_BASE_OFFSET = np.array([6.0, 0.0, 0.0])     

# Timing Phases (Frames)
PHASE_1 = 30
PHASE_2 = 50
PHASE_3 = 160 
PHASE_4 = 100 
TOTAL_FRAMES = PHASE_1 + PHASE_2 + PHASE_3 + PHASE_4

CUBE_SIZE = 4

# --- GEOMETRY & MATH FUNCTIONS ---

def get_current_axis_pos(t):
    """Calculates the current position of the precessing axis."""
    angle = AXIS_ORBIT_SPEED * t
    x = AXIS_BASE_OFFSET[0] * np.cos(angle) - AXIS_BASE_OFFSET[1] * np.sin(angle)
    y = AXIS_BASE_OFFSET[0] * np.sin(angle) + AXIS_BASE_OFFSET[1] * np.cos(angle)
    return np.array([x, y, 0.0])

def get_m4_cage(t):
    """Generates the wireframe containment cage."""
    scale = 1.5 + (EXPANSION_RATE * t * 0.4)
    angle = ROTATION_SPEED * t * 0.7
    grid_range = np.linspace(-15, 15, 8) 
    X, Y = np.meshgrid(grid_range, grid_range)
    X_rot = X * np.cos(angle) - Y * np.sin(angle)
    Y_rot = X * np.sin(angle) + Y * np.cos(angle)
    Z_top = np.full_like(X, 12.0 * scale)
    Z_bottom = np.full_like(X, -6.0 * scale)
    return X_rot * scale, Y_rot * scale, Z_top, Z_bottom

def get_chaotic_distortion(X, Y, Z, t, healing_factor=0.0):
    """Applies mathematically pseudo-random spatial distortion (The Ash)."""
    dist_mult = 1.0 - healing_factor
    twist = 0.15 * np.sin(t * 0.1) * dist_mult
    X_d = X + twist * Z
    Y_d = Y - twist * Z
    wave_1 = np.sin(X * 0.5 + t * 0.2) * np.cos(Y * 0.5 + t * 0.1)
    wave_2 = np.sin(X * 1.5 - t * 0.3) * 0.3
    Z_d = Z + (wave_1 + wave_2) * 1.5 * dist_mult
    X_d += np.cos(Z * 2 + t * 0.2) * 0.4 * dist_mult
    return X_d, Y_d, Z_d

def get_transformed_voxels(t, phase_type, healing=0.0):
    """Calculates voxel coordinates based on current phase and time."""
    base_range = np.linspace(-CUBE_SIZE/2, CUBE_SIZE/2, CUBE_SIZE + 1)
    if phase_type == 'full':
        z_range = np.linspace(0, CUBE_SIZE, CUBE_SIZE + 1)
    else:
        z_range = np.linspace(0, 0.8, CUBE_SIZE + 1)

    Xg, Yg, Zg = np.meshgrid(base_range, base_range, z_range, indexing='ij')

    if t > 0:
        angle = ROTATION_SPEED * t * 0.2
        X_rot = Xg * np.cos(angle) - Yg * np.sin(angle)
        Y_rot = Xg * np.sin(angle) + Yg * np.cos(angle)
        drift = A3_DRIFT_SPEED * t
        X_final, Y_final, Z_final = get_chaotic_distortion(X_rot, Y_rot, Zg, t, healing)
        X_final += drift[0]
        Y_final += drift[1]
        Z_final += drift[2]
    else:
        X_final, Y_final, Z_final = Xg, Yg, Zg

    center_pos = np.array([np.mean(X_final), np.mean(Y_final), np.mean(Z_final)])
    return X_final, Y_final, Z_final, center_pos

def get_attractor_visuals(t):
    """Generates the visual elements for the precessing attractor."""
    current_axis = get_current_axis_pos(t)
    z_line = np.linspace(-6, 18, 50)
    spin = t * 0.2
    r_helix = 1.5
    
    x_h1 = current_axis[0] + r_helix * np.cos(z_line * 0.5 + spin)
    y_h1 = current_axis[1] + r_helix * np.sin(z_line * 0.5 + spin)
    x_h2 = current_axis[0] + r_helix * np.cos(z_line * 0.5 + spin + np.pi)
    y_h2 = current_axis[1] + r_helix * np.sin(z_line * 0.5 + spin + np.pi)
    
    arrow_z = np.linspace(0, 14, 4)
    arrow_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    ax_x, ax_y, ax_z, u, v, w = [], [], [], [], [], []
    pulse = np.sin(t * 0.1) 
    
    for z in arrow_z:
        for ang in arrow_angles:
            r_start = 3.0
            cur_ang = ang + spin * 0.5
            sx = current_axis[0] + r_start * np.cos(cur_ang)
            sy = current_axis[1] + r_start * np.sin(cur_ang)
            sz = z
            dir_x = np.cos(cur_ang) * pulse
            dir_y = np.sin(cur_ang) * pulse
            dir_z = 0.2
            ax_x.append(sx)
            ax_y.append(sy)
            ax_z.append(sz)
            u.append(dir_x)
            v.append(dir_y)
            w.append(dir_z)
            
    return (x_h1, y_h1, z_line), (x_h2, y_h2, z_line), (ax_x, ax_y, ax_z, u, v, w), pulse, current_axis

def operator_trajectory_dynamic(t_current, t_total, start_pos, drift_start_time):
    """Calculates the smooth quadratic Bezier trajectory for the Operator."""
    progress = t_current / t_total
    future_time = drift_start_time + t_total 
    future_axis = get_current_axis_pos(future_time)
    target_pos = future_axis + np.array([0.0, 0.0, 14.0])
    control_point = (start_pos + target_pos) / 2 + np.array([0, 0, 5.0])
    p1 = (1-progress)**2 * start_pos
    p2 = 2 * (1-progress) * progress * control_point
    p3 = progress**2 * target_pos
    return p1 + p2 + p3

# --- INITIALIZATION ---
fig = plt.figure(figsize=(12, 10), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')

def update(frame):
    ax.cla()
    cam_limit = 14 + frame * 0.01
    ax.set_xlim([-cam_limit, cam_limit])
    ax.set_ylim([-cam_limit, cam_limit])
    ax.set_zlim([-8, 20])
    ax.axis('off')
    
    current_phase = ""
    op_pos = None
    
    voxel_bool = np.zeros((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), dtype=bool)
    voxel_colors = np.empty(voxel_bool.shape, dtype=object) 
    X_v, Y_v, Z_v = None, None, None

    # === PHASE LOGIC ===

    # 1. STABILITY
    if frame < PHASE_1:
        current_phase = f"PHASE I: DATA CUBE A1\nStatus: Stable Structure"
        voxel_bool[:, :, :] = True
        voxel_colors[:, :, :] = '#1f77b490'
        X_v, Y_v, Z_v, _ = get_transformed_voxels(0, 'full', healing=0.0)

    # 2. BIFURCATION (SVD Splitting)
    elif frame < PHASE_1 + PHASE_2:
        current_phase = "PHASE II: BIFURCATION\nExtracting Spectral Operator..."
        t = (frame - PHASE_1) / PHASE_2
        layers_left = int(CUBE_SIZE * (1.0 - t))
        if layers_left < 1: layers_left = 1
        
        voxel_bool[:, :, :layers_left] = True
        voxel_colors[:, :, :] = '#17becf80'
        X_v, Y_v, Z_v, _ = get_transformed_voxels(0, 'full', healing=0.0)
        op_pos = np.array([0.0, 0.0, CUBE_SIZE + t * 8.0])

    # 3. PRECESSING ATTRACTOR (Data Drift & Chaos)
    elif frame < PHASE_1 + PHASE_2 + PHASE_3:
        current_phase = "PHASE III: PRECESSING ATTRACTOR\nApplying Chaotic Permutations (The Ash)..."
        t_drift = frame - (PHASE_1 + PHASE_2)
        
        voxel_bool[:, :, 0] = True
        voxel_colors[:, :, 0] = '#00ffff60'
        X_v, Y_v, Z_v, a3_center_now = get_transformed_voxels(t_drift, 'flat', healing=0.0)
        
        start_op = np.array([0.0, 0.0, CUBE_SIZE + 8.0])
        op_pos = operator_trajectory_dynamic(t_drift, PHASE_3, start_op, PHASE_1 + PHASE_2)

    # 4. INTEGRATION & HEALING (Reconstruction)
    else:
        current_phase = "PHASE IV: INTEGRATION & HEALING\nReversing Entropy..."
        t_rest = (frame - (TOTAL_FRAMES - PHASE_4)) / PHASE_4
        t_total_drift = frame - (PHASE_1 + PHASE_2)
        
        voxel_bool[:, :, 0] = True 
        X_v, Y_v, Z_v, a3_center_now = get_transformed_voxels(t_total_drift, 'flat', healing=0.0)
        
        current_axis = get_current_axis_pos(frame)
        op_start_rest = current_axis + np.array([0.0, 0.0, 14.0])
        op_end_rest = a3_center_now 
        
        fly_time = 0.3
        if t_rest < fly_time:
             t_fly = t_rest / fly_time
             op_pos = op_start_rest * (1 - t_fly) + op_end_rest * t_fly
             voxel_colors[:, :, 0] = '#00ffff60'
        else:
             op_pos = op_end_rest 
             t_grow = (t_rest - fly_time) / (1.0 - fly_time)
             
             if t_grow > 0.98: t_grow = 1.0
             current_healing = t_grow 
             
             layers_to_restore = int(CUBE_SIZE * t_grow) + 1
             if layers_to_restore > CUBE_SIZE: layers_to_restore = CUBE_SIZE
             
             X_full, Y_full, Z_full, _ = get_transformed_voxels(t_total_drift, 'full', healing=current_healing)
             X_v, Y_v, Z_v = X_full, Y_full, Z_full
             
             voxel_bool[:, :, :layers_to_restore] = True
             voxel_colors[:, :, :] = '#1f77b490'
             
             # Calculate percentage for UI text
             integrity_pct = int(current_healing * 100)
             if integrity_pct >= 99: integrity_pct = 100
             
             current_phase += f"\nData Integrity: {integrity_pct}% | MSE = 0.0"

    # === RENDERING ===
    if frame > PHASE_1:
        Xm, Ym, Zmt, Zmb = get_m4_cage(frame - PHASE_1)
        ax.plot_wireframe(Xm, Ym, Zmt, color='magenta', alpha=0.1)
        ax.plot_wireframe(Xm, Ym, Zmb, color='magenta', alpha=0.1)

    if frame > PHASE_1 + PHASE_2:
        helix1, helix2, arrows, pulse, cur_ax = get_attractor_visuals(frame)
        
        ax.plot(helix1[0], helix1[1], helix1[2], color='yellow', linewidth=2, alpha=0.8)
        ax.plot(helix2[0], helix2[1], helix2[2], color='yellow', linewidth=2, alpha=0.8)
        ax.plot([cur_ax[0], cur_ax[0]], [cur_ax[1], cur_ax[1]], [-6, 18], 
                color='white', linestyle='--', alpha=0.3)
        
        col_arrow = 'lime' if pulse > 0 else 'red'
        ax.quiver(arrows[0], arrows[1], arrows[2], 
                  arrows[3], arrows[4], arrows[5], 
                  length=1.5, color=col_arrow, alpha=0.6, normalize=True)

    if X_v is not None:
        ax.voxels(X_v, Y_v, Z_v, voxel_bool, facecolors=voxel_colors, edgecolor='#00000030', linewidth=0.3, shade=True)

    if op_pos is not None:
        col_op = 'red'
        if "Integrity" in current_phase: col_op = '#ff00ff'
        ax.scatter(op_pos[0], op_pos[1], op_pos[2], color=col_op, s=300, marker='*', edgecolors='white', zorder=100)

    ax.text2D(0.05, 0.92, current_phase, transform=ax.transAxes, color='white', fontsize=11, fontfamily='monospace', weight='bold')

ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=40, repeat=False)
plt.show()