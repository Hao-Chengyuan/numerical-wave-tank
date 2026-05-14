from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

OUTPUT_DIR = Path(__file__).parent.parent / "output"

class RegularWave:
    """
    A class to represent a First-Order (Airy) Regular Wave.
    """
    def __init__(self, H, T, d, phase=0.0):

        if H <= 0 or T <= 0 or d <= 0:
            raise ValueError("Wave height, period, and water depth must be positive.")
        
        self.H = H                # Wave height [m]
        self.T = T                # Wave period [s]
        self.d = d                # Water depth [m]
        self.phase = phase        # Phase angle [rad]
        self.g = 9.81             # Gravity [m/s^2]
        
        self.omega = 2 * np.pi / self.T  # Angular frequency [rad/s]
        
        # Calculate wave number and derived properties upon initialization
        self.k = self._solve_dispersion()
        self.L = 2 * np.pi / self.k      # Wavelength [m]
        self.C = self.omega / self.k     # Wave celerity [m/s]
    
    def _solve_dispersion(self, tol=1e-6, max_iter=100, *, initial_guess: str = 'hunt'):
        """
        Solves the linear dispersion relation using the Newton-Raphson method.
        f(k) = g * k * tanh(kd) - omega^2 = 0
        """
        if initial_guess == 'hunt':
            # Hunt (1979) approximation for initial guess
            y = (self.omega**2 * self.d) / self.g
            kd_squared = y**2 + y / (1.0 + 0.6522*y + 0.4622*(y**2) + 0.0864*(y**4) + 0.0675*(y**5))
            k_guess = np.sqrt(kd_squared) / self.d
        else:
            # Initial guess assuming deep water: tanh(kd) -> 1
            k_guess = self.omega**2 / self.g
        
        for _ in range(max_iter):
            # The function: f(k)
            f_k = self.g * k_guess * np.tanh(k_guess * self.d) - self.omega**2
            
            # The derivative: f'(k)
            # Derivative of tanh(x) is sech^2(x) = 1 - tanh^2(x)
            f_prime_k = self.g * np.tanh(k_guess * self.d) + \
                        self.g * k_guess * self.d * (1.0 - np.tanh(k_guess * self.d)**2)
            
            # Newton-Raphson step
            k_new = k_guess - (f_k / f_prime_k)
            
            # Check for convergence
            if abs(k_new - k_guess) < tol:
                return k_new
                
            k_guess = k_new
            
        raise ValueError(f"Dispersion relation failed to converge after {max_iter} iterations.")
    
    def surface_elevation(self, x, t):
        """
        Calculates the free surface elevation eta(x, t).
        x and t can be floats or numpy arrays.
        """
        amplitude = self.H / 2.0
        return amplitude * np.cos(self.k * x - self.omega * t + self.phase)
    

class IrregularWave_JONSWAP:
    """
    Generates a 1D irregular sea state from a JONSWAP spectrum using 
    statistically correct Rayleigh-distributed amplitudes.
    """
    def __init__(self, Hs, Tp, d, gamma=3.3, T_repeat=10800, seed=None):
        self.Hs = Hs
        self.Tp = Tp
        self.d = d
        self.gamma = gamma
        
        # Set up the frequency domain
        fp = 1.0 / Tp
        df = 1.0 / T_repeat
        self.f = np.arange(0.3 * fp, 3.0 * fp, df)
        
        if seed is not None:
            np.random.seed(seed)
            
        # 1. Generate the JONSWAP Spectrum
        S_f = self._jonswap()
        
        # 2. Generate Random Amplitudes (Rayleigh) and Phases (Uniform)
        scale_param = np.sqrt(S_f * df)
        self.amplitudes = np.random.rayleigh(scale=scale_param)
        self.phases = np.random.uniform(-np.pi, np.pi, len(self.f))
        
        # 3. Instantiate the RegularWave objects
        self.wave_components = []
        for i in range(len(self.f)):
            # H = 2 * Amplitude
            H_comp = 2.0 * self.amplitudes[i]
            T_comp = 1.0 / self.f[i]
            
            wave = RegularWave(H=H_comp, T=T_comp, d=self.d, phase=self.phases[i])
            self.wave_components.append(wave)
            
    def _jonswap(self):
        fp = 1.0 / self.Tp
        sigma = np.where(self.f <= fp, 0.07, 0.09)
        S_pm = (5.0 / 16.0) * (self.Hs**2) * (fp**4) * (self.f**-5) * np.exp(-1.25 * (self.f / fp)**-4)
        J = self.gamma ** np.exp(-0.5 * ((self.f - fp) / (sigma * fp))**2)
        A_gamma = 1 - 0.287 * np.log(self.gamma)
        return A_gamma * S_pm * J

    def surface_elevation(self, x, t):
        """
        Calculates the total surface elevation by summing all component waves.
        """
        eta_total = 0.0
        for wave in self.wave_components:
            eta_total += wave.surface_elevation(x, t)
        return eta_total


if __name__ == "__main__":
    # Create a regular wave: H = 2m, T = 8s, depth = 50m
    wave = RegularWave(H=2.0, T=8.0, d=50.0)
    
    print(f"Wave Period: {wave.T} s")
    print(f"Wavelength:  {wave.L:.2f} m")
    print(f"Celerity:    {wave.C:.2f} m/s")
    
    # Let's plot a spatial snapshot of this wave at t = 0
    x_domain = np.linspace(0, 3 * wave.L, 500) # Plot 3 wavelengths
    eta_spatial = wave.surface_elevation(x=x_domain, t=0)
    
    plt.figure(figsize=(10, 4))
    plt.plot(x_domain, eta_spatial, 'b-')
    plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
    plt.title(f"Regular Wave Spatial Snapshot (t=0s)\nH={wave.H}m, L={wave.L:.1f}m, d={wave.d}m")
    plt.xlabel("Distance x [m]")
    plt.ylabel(r"Elevation $\eta$ [m]")
    plt.grid(True)
    plt.tight_layout()

    # Create a sea state in 50m of water
    storm = IrregularWave_JONSWAP(Hs=2.5, Tp=10.0, d=50.0, seed=42)
    
    # Snapshots of the irregular wave field:
    # 1. Plot a Time Series at x = 0
    t_domain = np.linspace(0, 100, 500) # 100 seconds
    eta_time = storm.surface_elevation(x=0.0, t=t_domain)
    
    # 2. Plot a Spatial Snapshot at t = 0
    x_domain = np.linspace(0, 300, 500) # 300 meters long
    eta_space = storm.surface_elevation(x=x_domain, t=0.0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(t_domain, eta_time, 'k-')
    ax1.set_title("Time Series at Fixed Point (x=0 m)")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel(r"Elevation $\eta$ [m]")
    ax1.grid(True)
    
    ax2.plot(x_domain, eta_space, 'b-')
    ax2.set_title("Spatial Snapshot across the Wave Tank (t=0 s)")
    ax2.set_xlabel("Distance [m]")
    ax2.set_ylabel(r"Elevation $\eta$ [m]")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

    # Animate the irregular wave field over time
    x_domain = np.linspace(0, 300, 500)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 300)
    ax.set_ylim(-3, 3)
    ax.set_title("Irregular Sea State Surface Elevation")
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Elevation [m]")
    ax.grid(True)
    ax.axhline(0, color='black', linewidth=0.5)

    line, = ax.plot([], [], 'b-', lw=2)

    def init():
        line.set_data([], [])
        return (line,)
    
    def animate(frame):
        """
        This function is called for each frame. 
        'frame' is an integer that increments automatically.
        """
        # Calculate current time. Let's say each frame is 0.1 seconds.
        current_time = frame * 0.1 
        
        # Calculate new elevations for the entire spatial array at this time
        eta = storm.surface_elevation(x_domain, current_time)
        
        # Update the line data
        line.set_data(x_domain, eta)
        return (line,)

    # Create the animation object
    # frames=200 means we calculate 200 time steps. At 0.1s/step, this is a 20s animation.
    # interval=50 is the delay between frames in milliseconds when displaying it live.
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=200, 
                                interval=50, blit=True)
    
    print("Generating animation. This might take a minute...")

    gif_path = OUTPUT_DIR / "Irregular_Wave_Tank.gif"
    ani.save(str(gif_path), writer='pillow', fps=20)
    print(f"Animation successfully saved as: {gif_path}")

    # Note: If you want to save it as an .mp4, you need to have FFmpeg installed on your machine.
    # If you do have it, you can uncomment the lines below:
    # mp4_filename = "Irregular_Wave_Tank.mp4"
    # ani.save(mp4_filename, writer='ffmpeg', fps=20)
    # print(f"Animation successfully saved as: {mp4_filename}")

    plt.show() # Uncomment if you also want to see the live preview window