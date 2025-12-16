import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Default parameters for reference
# Paramètres des voitures
N_ref = 50                      # Nombre de voitures
L_ref = 4.5                    # Longueur d'une voiture 
spacing_ref = 50.0              # Espacement de départ

# Paramètres Physiques
V_target_ref = 27.0             # Vitesse qu'on veut atteindre
a_max_ref = 3.0                # Accelation max
a_min_ref = -7.0               # Décélération max

# Paramètres de Comportement
k_ref = 0.3                     # Agressivité en accélération
tau_ref = 1.0                    # Temps de reponse (temps de réponse en s)
ttc_ref = 2.0                  # Time to collision (s)
# # Paramètres Temporels
dt = 0.05                       # Pas de temps
T_total = 300.0                 # Durée totale

# Define a structure for simulation parameters
def create_params(N=N_ref, L=L_ref, spacing=spacing_ref, V_target=V_target_ref, t_reac=tau_ref, a_max=a_max_ref, a_min = a_min_ref, k=k_ref, ttc=ttc_ref):
    """Create a dictionary of simulation parameters. Defaults are set to reference values."""
    
    if not isinstance(V_target, np.ndarray):
        V_target = np.ones(N) * V_target
    if not isinstance(t_reac, np.ndarray):
        t_reac = np.ones(N, np.float32) * t_reac
    if not isinstance(ttc, np.ndarray):
        ttc = np.ones(N, dtype=np.float32) * ttc
    if not isinstance(k, np.ndarray):
        k = np.ones(N, dtype=np.float32) * k

    params = {
        'N': N,
        'L': L,
        'spacing': spacing,
        'V_target': V_target,
        't_reac': t_reac,
        'a_max': a_max,
        'a_min': a_min,
        'k': k,
        'ttc': ttc,
    }
    return params

def a_adjust(vitesses, positions, params):
    """Compute acceleration adjustments for vehicles based on their velocities and positions."""
    u = np.zeros(params['N'])
    u[0] = params['k'][0] * (params['V_target'][0] - vitesses[0])

    # Calculate current gap and desired gap
    alpha = (positions[:-1]- positions[1:]) - params['L']

    # Calculate optimal braking distance
    braking_distance = (vitesses[1:]**2) / (2 * abs(params['a_min']))
    
    # Calculate desired distance 
    alpha_want = np.maximum(vitesses[1:] * params['ttc'][1:] + braking_distance * (1-params['k'][1:]), params['L']*2)


    l = np.clip(alpha / alpha_want, 0.0, 1.0)

    a_norm = np.minimum(params['k'][1:] * (params['V_target'][1:] - vitesses[1:]), params['a_max'])
    a_crit = params['a_min']
    # Interpolation 
    u[1:] = l * a_norm + (1 - l) * a_crit
    return u

def simulate(params, brake_moments, metrics=False):
    """Simulate the traffic flow based on given parameters and brake moments.
    Arguments:
    - params: Dictionary containing simulation parameters.
    - brake_moments: List of tuples (moment, duration, force) indicating when brakes are applied to the leading car
    Returns:
    - X: Positions of cars over time.
    - V: Velocities of cars over time.
    - crash: Tuple (time, position) if an accident occurs, else None.
    """
    nb_steps = int(T_total / dt)

    # Extract parameters
    N = params['N']
    L = params['L']
    spacing = params['spacing']
    V_target = params['V_target']
    k = params['k']

    X = np.zeros((nb_steps, N))
    V = np.zeros((nb_steps, N))
    A = np.zeros((nb_steps, N))

    # Initial positions
    gaps = V_target * params['ttc'] + L


    # Make the distances initially safe, generate X from gaps
    X[0] = -np.cumsum(np.concatenate(([0], gaps[:-1])))
    # X[0] = -np.arange(N) * spacing
    # X[0] = -np.arange(N) * spacing
    V[0] = V_target

    crash = None

    for i in range(nb_steps - 1):
        t = i * dt

        # Check for brake
        perturb = 0
        for moment, duration, force in brake_moments:
            if moment < t < moment + duration:
                perturb = force
                break
        # perturb = (10.0 < t < 13.0)

        # get acceleration
        u = a_adjust(V[i], X[i], params) 
            
        if perturb>0:
            u[0] = -perturb

        # step  
        A[i+1] = A[i] + (dt / params['t_reac']) * (u - A[i])
        V[i+1] = np.maximum(V[i] + A[i+1] * dt, 0.0)
        X[i+1] = X[i] + V[i+1] * dt
        

        # Check for accidents
        alphas = (X[i+1, :-1] - X[i+1, 1:]) - L


        if np.any(alphas < 0):
            if crash is None:
                idx_crash = np.where(alphas < 0)[0][0] + 1
                pos_crash = X[i+1, idx_crash]
                crash = (t, pos_crash)

            #Stop simulation
            X[i+1:] = X[i+1]
            V[i+1:] = V[i+1]
            break
        
    if metrics:
        gaps = (X[-1, :-1] - X[-1, 1:]) - L
        
        speed_std = np.std(V[-1])       # Speed instability
        speed_mean = np.mean(V[-1])     # Traffic flow efficiency
        
        gap_std = np.std(gaps)          # Spacing instability (bunching)
        gap_mean = np.mean(gaps)        # Average density indicator
        gap_min = np.min(gaps)          # Critical safety margin
        
        return np.array([float(int(crash is not None)), speed_std, speed_mean, gap_std, gap_mean, gap_min])
    
        # avg_stability, max_stability
        # return speed_std, speed_mean, gap_std, gap_mean, gap_min, crash is not None
    return X, V, crash

def get_simulation_data(params, brake, brake_moments):
    """Call simulation with given parameters and brake settings."""
    brake_force, brake_time = brake
    if brake_moments is None:
        brake_moments = []
    brakes = [(m, brake_time, brake_force) for m in brake_moments]
    return simulate(params, brakes)

def visualize_trajectories(params, brake_force, ax):

    X, V, crash_info = simulate(params, [(10, 4, brake_force)])

    time_grid = np.arange(len(X)) * dt

    for i in range(0, params['N']):
        ax.plot(time_grid, X[:, i], color='black', alpha=0.3, linewidth=1)

    if crash_info:
        c_time, c_loc = crash_info
        ax.scatter(c_time, c_loc, c='red', s=100, marker='X', zorder=5)
        ax.set_title(f"Simulation Frozen t={c_time:.2f}s (CRASH)", color='red', fontweight='bold')
        ax.axvspan(c_time, T_total, color='red', alpha=0.1)
    else:
        ax.set_title("Trajectories", color='green', fontweight='bold')

    ax.set_ylabel("Position (m)")
    ax.set_xlabel("Time (s)")

def visualize(data, params, brake_moments, t, return_data=False):
    """Visualize the traffic simulation at a specific time t.
        3 Plots are created:
        - Positions of cars on the road
        - Speeds of cars
        - Distances between cars
    Arguments:
    - data: Tuple (X, V, crash_info) from simulation.
    - params: Dictionary of simulation parameters.
    - brake_moments: List of times when brakes were applied.
    - t: Time at which to visualize the state.
    - return_data: If True, return the figure and axes instead of displaying.

    Returns:
    - If return_data is True, returns (fig, axes, plot_elements, data)
    - Else, displays the plot.
    """

    X, V, crash_info = data
    
    if brake_moments is None:
        brake_moments = []
    
    crash_moment = crash_info[0] if crash_info else T_total
    brake_points = X[[int(m / dt) for m in  [m for m in brake_moments if m < crash_moment]], 0]
    road_min = -params['N'] * params['spacing'] -100
    road_max = np.max(params['V_target']) * T_total +100
    
    # Get State
    t_idx = int(t / dt)
    if t_idx >= len(X): t_idx = len(X) - 1
    x_now = X[t_idx]
    v_now = V[t_idx]
    

    fig, (ax_pos, ax_vel, ax_alphas) = plt.subplots(3, 1, figsize=(16, 12))
    
    # Position Plot (ax_pos)
    ax_pos.hlines(0, road_min, road_max, colors='gray', linestyles='-', linewidth=2, alpha=0.3)
    ax_pos.vlines(brake_points, color='orange', linestyle='--', linewidth=2, label='Braking Point', ymin=-3, ymax=3)
    
    # Draw Crash
    if crash_info:
        c_time, c_loc = crash_info
        if t >= c_time:
            ax_pos.axvline(c_loc, color='red', linestyle=':', linewidth=3, label='CRASH POINT')
            ax_pos.text(c_loc, 1.2, "IMPACT", color='red', fontweight='bold', ha='center')

    # Color mapping
    norm = plt.Normalize(0, 35)
    cmap = plt.cm.RdYlGn
    colors = cmap(norm(v_now))
    
    # Crash Coloring
    if params['N'] > 1:
        gaps = x_now[:-1] - x_now[1:] - params['L']
        crashed_indices = np.where(gaps < 0)[0] + 1
        if len(crashed_indices) > 0:
            colors[crashed_indices] = np.array([1.0, 0.0, 0.0, 1.0])

    # Draw Cars
    sc = ax_pos.scatter(x_now, np.zeros_like(x_now), c=colors, s=30, edgecolors='black', zorder=3)
    
    ax_pos.set_title(f"Traffic State (t={t:.1f}s)", fontsize=14)
    ax_pos.set_ylim(-1, 1)
    ax_pos.set_yticks([])
    ax_pos.set_xlabel("Road Position (m)")
    ax_pos.set_xlim(road_min, road_max)
    ax_pos.legend(loc='upper right')
    
    # Speeds (ax_vel)
    bars_vel = ax_vel.bar(-np.arange(len(v_now)), v_now, width=.5, color=cmap(norm(v_now)), alpha=0.8, align='center')
    ax_vel.set_ylabel("Speed")
    ax_vel.set_ylim(0, 35)


    dists = np.zeros(2*len(x_now)-1)
    dists[1::2] =  x_now[:-1] - x_now[1:]

    bars_alpha = ax_alphas.bar(-np.arange(2*len(x_now)-1),dists, width=1, alpha=0.8, align='center')
    
    if return_data:
        plt.close(fig)
        return fig, (ax_pos, ax_vel, ax_alphas), (sc, bars_vel, bars_alpha), (X, V, crash_info)

    plt.tight_layout()
    plt.show()

def animate(params, brake, brake_moments=None):
    """Create an animation of the traffic simulation.
    Arguments:
    - params: Dictionary of simulation parameters.
    - brake: Tuple (brake_force, brake_time).
    - brake_moments: List of times when brakes are applied.
    Returns:
    - anim: Matplotlib animation object.
    """
    
    data = get_simulation_data(params, brake, brake_moments)
    fig, (ax_pos, _, _), (sc, bars_vel, bars_alpha), (X, V, _) = visualize(data, params, brake_moments, 0, return_data=True)
    
    norm = plt.Normalize(0, 35)
    cmap = plt.cm.RdYlGn

    def update(frame):
        x, v = X[frame], V[frame]
        
        # Update positions and colors
        colors = cmap(norm(v))
        if params['N'] > 1:
            gaps = x[:-1] - x[1:] - params['L']
            crashed = np.where(gaps < 0)[0] + 1
            if len(crashed) > 0: colors[crashed] = [1, 0, 0, 1]
            
        sc.set_offsets(np.c_[x, np.zeros_like(x)])
        sc.set_facecolors(colors)
        
        # Update bars
        for bar, h, c in zip(bars_vel, v, colors):
            bar.set_height(h)
            bar.set_color(c)
            
        dists = np.zeros(2*len(x)-1)
        dists[1::2] = x[:-1] - x[1:]
        for bar, h in zip(bars_alpha, dists):
            bar.set_height(h)
            
        # ax_pos.set_title(, fontsize=14)
        # ax_pos.set_title(
        #     f"Traffic State (t={frame*dt:.1f}s) \n"
        #     f"V_target={params['V_target']:.1f} m/s | "
        #     f"t_reac={params['t_reac']:.2f}s | "
        #     f"ttc={params['ttc']:.1f}s | "
        #     f"brake={brake[0]:.1f} for {brake[1]:.1f}s"
        # )
        return [sc, *bars_vel, *bars_alpha]

    anim = animation.FuncAnimation(fig, update, frames=range(0, len(X), 50), interval=200, blit=False)

    return anim
