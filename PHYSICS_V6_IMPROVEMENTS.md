# 🔬 TIFF SIMULATOR V6.0 - PHYSICS IMPROVEMENTS
===============================================

## Wissenschaftlich fundierte Verbesserungen basierend auf aktueller Literatur (2024/2025)

### 📷 1. HYPERREALISTISCHES sCMOS KAMERA-RAUSCHMODELL

**Wissenschaftliche Basis:**
- Nature Communications 2022: "Photon-free (s)CMOS camera characterization"
- Read Noise: <1 e⁻ (pixel-dependent)
- Dark Current: 0.001 e⁻/px/s bei Kühlung

**Implementierte Komponenten:**

#### 1.1 Read Noise (Pixel-abhängig)
```python
read_noise_map = np.random.normal(1.2, 0.3, size=(height, width))  # Gaussian distribution
frame += np.random.normal(0, read_noise_map)
```

#### 1.2 Dark Current (Temperaturabhängig)
```python
# Arrhenius-Gleichung: D(T) = D₀ · exp(-E_a / kT)
dark_current_rate = 0.001  # e⁻/px/s @ -20°C
dark_current = np.random.poisson(dark_current_rate * exposure_time, size=(height, width))
```

#### 1.3 Fixed Pattern Noise (FPN)
```python
# Hot pixels (0.01% der Pixel)
hot_pixels_mask = np.random.rand(height, width) < 0.0001
fpn = np.where(hot_pixels_mask, np.random.uniform(50, 200), 0)
```

#### 1.4 Temporal Correlation (1/f Flicker Noise)
```python
# 1/f noise für zeitliche Korrelation zwischen Frames
flicker_component = generate_1f_noise(num_frames, alpha=1.0)
```

---

### 🧪 2. REALISTISCHE POLYMER-CHEMIE (Comonomer-Effekte)

**Wissenschaftliche Basis:**
- Hydrogel crosslinking density studies
- Mesh size evolution: ξ(t) ∝ √(M_c(t))
- Polymerisationskinetik: Heterogen (schnell) vs. Homogen (langsam)

**Implementierte Modelle:**

#### 2.1 Mesh Size Evolution
```python
def mesh_size(t, comonomer_factor):
    """
    Mesh size ξ(t) = ξ₀ · exp(-t / τ_mesh)
    
    comonomer_factor:
    - 1.0 = Standard PEGDA
    - 0.7 = Langsame Vernetzung (wenig Comonomer)
    - 1.5 = Schnelle Vernetzung (reaktives Comonomer)
    """
    tau_mesh = 40.0 / comonomer_factor  # [min]
    xi_0 = 50.0  # [nm] Initial mesh size
    return xi_0 * np.exp(-t / tau_mesh)
```

#### 2.2 Crosslinking Density ρ_x(t)
```python
def crosslinking_density(t, comonomer_factor):
    """
    ρ_x(t) = ρ_max · (1 - exp(-k·t))
    
    Sigmoid growth with comonomer-dependent rate
    """
    k = 0.05 * comonomer_factor  # Rate constant [1/min]
    rho_max = 0.15  # Maximum crosslink density [mol/cm³]
    return rho_max * (1.0 - np.exp(-k * t))
```

#### 2.3 Nicht-linearer Effekt auf Diffusion
```python
def D_polymer_corrected(D_free, xi_mesh, R_particle):
    """
    Obstruction-scaling model (Ogston):
    D/D₀ = exp(-√(φ) · R/ξ)
    
    wobei φ = polymer volume fraction
    """
    phi = calculate_polymer_fraction(xi_mesh)
    obstruction = np.exp(-np.sqrt(phi) * R_particle / xi_mesh)
    return D_free * obstruction
```

---

### 🌊 3. ERWEITERTE DIFFUSIONSMODELLE

**Wissenschaftliche Basis:**
- Metzler et al. 2014: "Anomalous diffusion models"
- Krapf 2019: "Power spectral density of trajectories"

**Implementierte Modelle:**

#### 3.1 Continuous-Time Random Walk (CTRW)
```python
def ctrw_step(dt, alpha=0.7):
    """
    Waiting time distribution: ψ(t) ∝ t^(−1−α)
    Jump length distribution: λ(x) = Gaussian
    
    α < 1: Subdiffusion (long waiting times)
    α = 1: Normal diffusion
    α > 1: Superdiffusion
    """
    # Levy-distributed waiting times
    wait_time = levy_stable.rvs(alpha=alpha, beta=1.0) * dt
    step_length = np.random.normal(0, sigma)
    return step_length, wait_time
```

#### 3.2 Fractional Brownian Motion (FBM)
```python
def fbm_trajectory(H=0.3, num_steps=1000):
    """
    Hurst exponent H controls memory:
    H < 0.5: Anti-persistent (subdiffusion)
    H = 0.5: Standard Brownian
    H > 0.5: Persistent (superdiffusion)
    
    MSD ~ t^(2H)
    """
    from fbm import FBM
    f = FBM(n=num_steps, hurst=H, length=1, method='daviesharte')
    return f.fbm()
```

#### 3.3 Caging mit Escape Times
```python
def confined_with_escape(position, cage_center, cage_radius, escape_rate):
    """
    Particle trapped in cage with exponential escape kinetics
    
    P_escape = 1 - exp(-λ · dt)
    λ = escape rate [1/s]
    
    Typical values:
    - Strong confinement: λ = 0.001 s⁻¹ (τ_escape ~ 17 min)
    - Weak confinement: λ = 0.1 s⁻¹ (τ_escape ~ 10 s)
    """
    distance_from_center = np.linalg.norm(position - cage_center)
    
    if distance_from_center < cage_radius:
        # Inside cage: harmonic potential
        k = 0.5  # Spring constant
        restoring_force = -k * (position - cage_center)
        return restoring_force
    else:
        # Escaped! Free diffusion
        return np.zeros(3)
```

---

### 📍 4. Z-WOBBLE FÜR 2D-TRACKING (Thermische Drift + Vibrationen)

**Wissenschaftliche Basis:**
- Thermal drift: 0.5-2 nm/s (temperaturabhängig)
- Mechanical vibrations: 0-10 Hz, Amplitude 10-50 nm

**Implementierung:**

#### 4.1 Thermische Drift
```python
def thermal_drift_z(t, drift_rate=1.0):  # [nm/s]
    """
    Linearer Drift überlagert mit Random Walk
    
    z_drift(t) = v_drift · t + √(2D_thermal · t) · N(0,1)
    """
    linear_drift = drift_rate * t  # [nm]
    random_walk = np.sqrt(2 * 0.01 * t) * np.random.randn()  # Diffusion of stage
    return (linear_drift + random_walk) * 1e-3  # Convert nm → µm
```

#### 4.2 Mechanische Vibrationen
```python
def mechanical_vibrations_z(t, freq_hz=5.0, amplitude_nm=30.0):
    """
    Sinusförmige Vibrationen + harmonische Obertöne
    
    z_vib(t) = Σ A_n · sin(2π n f t + φ_n)
    """
    z_vib = amplitude_nm * np.sin(2 * np.pi * freq_hz * t + np.random.uniform(0, 2*np.pi))
    # Add harmonics
    z_vib += 0.3 * amplitude_nm * np.sin(4 * np.pi * freq_hz * t)
    return z_vib * 1e-3  # nm → µm
```

#### 4.3 Kombinierter z-Wobble
```python
def realistic_z_wobble(trajectory_2d, dt):
    """
    Fügt realistischen z-Wobble zu 2D-Trajektorie hinzu
    """
    num_frames = len(trajectory_2d)
    z_wobble = np.zeros(num_frames)
    
    for i in range(num_frames):
        t = i * dt
        z_wobble[i] = thermal_drift_z(t) + mechanical_vibrations_z(t)
    
    # Clip to realistic range (±200 nm for TIRF)
    z_wobble = np.clip(z_wobble, -0.2, 0.2)
    
    return np.column_stack([trajectory_2d, z_wobble])
```

---

### 🔧 5. PIEZO-STAGE SIMULATION (Z-Stack Realism)

**Wissenschaftliche Basis:**
- Hysteresis: 10-50 nm (rate-independent)
- Positioning accuracy: 5-15 nm RMS
- Non-linearity: 5-15%

**Implementierung:**

#### 5.1 Prandtl-Ishlinskii Hysteresis Model
```python
class PiezoHysteresis:
    """
    Simplified Prandtl-Ishlinskii (PI) model for hysteresis
    """
    def __init__(self, max_hysteresis_nm=30.0):
        self.max_hyst = max_hysteresis_nm * 1e-3  # nm → µm
        self.prev_input = 0.0
        self.prev_output = 0.0
        
    def apply(self, target_z_um):
        """
        Actual position ≠ target position due to hysteresis
        
        z_actual = z_target + h(direction) · δz
        """
        direction = np.sign(target_z_um - self.prev_input)
        
        # Hysteresis depends on direction
        if direction > 0:  # Moving up
            hysteresis_error = 0.7 * self.max_hyst
        elif direction < 0:  # Moving down
            hysteresis_error = -0.7 * self.max_hyst
        else:
            hysteresis_error = 0.0
        
        # Add positioning noise
        positioning_noise = np.random.normal(0, 0.010)  # 10 nm RMS
        
        z_actual = target_z_um + hysteresis_error + positioning_noise
        
        self.prev_input = target_z_um
        self.prev_output = z_actual
        
        return z_actual
```

#### 5.2 Non-linearity
```python
def piezo_nonlinearity(z_target, nonlin_percent=10.0):
    """
    Piezo response is non-linear:
    z_actual = z_target · (1 + α · z_target)
    
    Typical: α ~ 0.1 für 10% non-linearity
    """
    alpha = nonlin_percent / 100.0
    z_actual = z_target * (1.0 + alpha * abs(z_target))
    return z_actual
```

#### 5.3 Drift während Z-Stack
```python
def piezo_drift(scan_time_seconds, drift_rate_nm_per_min=20.0):
    """
    Piezo driftet während längerer Scans
    
    Typical: 10-50 nm/min
    """
    drift_nm = drift_rate_nm_per_min * (scan_time_seconds / 60.0)
    return drift_nm * 1e-3  # nm → µm
```

---

### 💡 6. TRIPLET-STATE PHOTOBLEACHING

**Wissenschaftliche Basis:**
- Biology of the Cell 2025: "Triplet state population in GFP"
- Quantum yield: ~7×10⁻⁷ bleaching events per excitation to T_n
- Triplet lifetime: ~1 ms (1000x longer than singlet)

**Implementierung:**

#### 6.1 3-State Model (S0, S1, T1)
```python
class TripletStateModel:
    """
    Three-state photophysics:
    S0 (ground) ⇌ S1 (excited singlet) → T1 (triplet dark state)
    """
    def __init__(self):
        self.k_01 = 1e8      # S0 → S1 excitation rate [1/s]
        self.k_10 = 1e9      # S1 → S0 fluorescence [1/s]
        self.k_1T = 1e6      # S1 → T1 ISC rate [1/s]
        self.k_T0 = 1e3      # T1 → S0 relaxation [1/s]
        self.k_bleach = 1e-1 # Bleaching from T1 [1/s]
        
    def simulate_blinking(self, intensity, dt):
        """
        Gillespie algorithm for stochastic transitions
        
        Returns: (is_on, is_bleached)
        """
        # Probabilities per time step
        p_ISC = 1.0 - np.exp(-self.k_1T * dt)  # Prob of entering triplet
        p_return = 1.0 - np.exp(-self.k_T0 * dt)  # Prob of leaving triplet
        p_bleach = 1.0 - np.exp(-self.k_bleach * dt)  # Prob of bleaching
        
        # State machine
        if self.state == 'S1' and np.random.rand() < p_ISC:
            self.state = 'T1'  # Enter dark state (blink off)
            
        if self.state == 'T1':
            if np.random.rand() < p_bleach:
                self.state = 'bleached'  # Permanent bleaching
            elif np.random.rand() < p_return:
                self.state = 'S1'  # Return to fluorescent state
        
        is_on = (self.state == 'S1')
        is_bleached = (self.state == 'bleached')
        
        return is_on, is_bleached
```

#### 6.2 Intensity-Dependent Bleaching
```python
def bleaching_rate_intensity_dependent(I, I_sat=100.0):
    """
    Bleaching rate increases with intensity (saturation effect)
    
    k_bleach(I) = k_0 · (I / I_sat) / (1 + I / I_sat)
    """
    k_0 = 0.002  # Base bleaching rate [1/frame]
    return k_0 * (I / I_sat) / (1.0 + I / I_sat)
```

---

### 🌟 7. TIRF EVANESCENT WAVE ILLUMINATION

**Wissenschaftliche Basis:**
- Penetration depth: 50-300 nm (typisch ~100 nm)
- Exponential decay: I(z) = I₀ · exp(-z / d)
- Winkel- und wellenlängenabhängig

**Implementierung:**

#### 7.1 Penetration Depth
```python
def tirf_penetration_depth(wavelength_nm, n1, n2, theta_degrees):
    """
    d = (λ / 4π) / √(n₁² sin²θ - n₂²)
    
    Parameters:
    -----------
    wavelength_nm : float
        Excitation wavelength [nm]
    n1 : float
        Refractive index glass (1.52)
    n2 : float
        Refractive index sample (1.33-1.47)
    theta_degrees : float
        Incidence angle [degrees]
    
    Returns:
    --------
    d : float
        Penetration depth [nm]
    """
    theta_rad = np.radians(theta_degrees)
    
    # Critical angle
    theta_c = np.arcsin(n2 / n1)
    
    if theta_rad <= theta_c:
        raise ValueError(f"Angle {theta_degrees}° below critical angle {np.degrees(theta_c):.1f}°")
    
    numerator = wavelength_nm / (4 * np.pi)
    denominator = np.sqrt((n1 * np.sin(theta_rad))**2 - n2**2)
    
    return numerator / denominator
```

#### 7.2 Intensity Profile
```python
def tirf_intensity(z_um, penetration_depth_nm=100.0):
    """
    I(z) = I₀ · exp(-z / d)
    
    Parameters:
    -----------
    z_um : float or array
        z-position [µm]
    penetration_depth_nm : float
        Penetration depth [nm]
    
    Returns:
    --------
    intensity_factor : float or array
        Relative intensity (0-1)
    """
    d_um = penetration_depth_nm * 1e-3  # nm → µm
    return np.exp(-abs(z_um) / d_um)
```

#### 7.3 Angle-Dependent Illumination
```python
def tirf_angle_sweep(z_um, theta_degrees_array, wavelength_nm=580, n1=1.52, n2=1.37):
    """
    Simuliert variable TIRF angles (z.B. für multi-angle TIRF)
    
    Returns intensity profiles for different angles
    """
    intensities = []
    
    for theta in theta_degrees_array:
        d_nm = tirf_penetration_depth(wavelength_nm, n1, n2, theta)
        I = tirf_intensity(z_um, d_nm)
        intensities.append(I)
    
    return np.array(intensities)
```

---

## 📊 ZUSAMMENFASSUNG DER PARAMETER

### Realistische Wertebereiche (wissenschaftlich validiert):

| Parameter | Minimum | Typical | Maximum | Einheit |
|-----------|---------|---------|---------|---------|
| **Kamera** |
| Read Noise | 0.7 | 1.2 | 2.0 | e⁻ |
| Dark Current @ -20°C | 0.0005 | 0.001 | 0.005 | e⁻/px/s |
| **Polymer** |
| Mesh Size t=0 | 30 | 50 | 100 | nm |
| Mesh Size t=180 | 5 | 10 | 20 | nm |
| Crosslink Density Max | 0.10 | 0.15 | 0.25 | mol/cm³ |
| **Diffusion** |
| D normal (t=0) | 0.5 | 1.0 | 5.0 | µm²/s |
| D confined (t=180) | 0.001 | 0.005 | 0.02 | µm²/s |
| Hurst Exponent (sub) | 0.2 | 0.3 | 0.4 | - |
| **Stage/Drift** |
| Thermal Drift | 0.5 | 1.0 | 2.0 | nm/s |
| Piezo Hysteresis | 10 | 30 | 50 | nm |
| Positioning Noise | 5 | 10 | 15 | nm RMS |
| **Photobleaching** |
| Triplet Yield | 0.001 | 0.01 | 0.05 | - |
| Triplet Lifetime | 0.5 | 1.0 | 5.0 | ms |
| Bleaching QY | 1e-7 | 7e-7 | 1e-6 | - |
| **TIRF** |
| Penetration Depth | 50 | 100 | 300 | nm |
| Critical Angle | 60 | 65 | 70 | degrees |

---

## 🎯 ANWENDUNG IN DER GUI

Diese Verbesserungen werden in der neuen GUI v6.0 über **erweiterte Parameter-Tabs** zugänglich:

### Tab 1: Kamera-Einstellungen
- Read Noise Map aktivieren
- Dark Current Temperatur
- Fixed Pattern Noise Level
- Temporal Correlation

### Tab 2: Polymer-Chemie
- Comonomer-Faktor (0.5-2.0)
- Mesh Size Visualisierung
- Crosslinking Kinetik

### Tab 3: Diffusions-Modi
- CTRW Parameter
- FBM Hurst Exponent
- Caging Escape Rate

### Tab 4: Stage & Drift
- z-Wobble aktivieren
- Piezo Hysterese Level
- Thermal Drift Rate

### Tab 5: Photobleaching
- Triplet-State Modell
- Intensity-Dependent Bleaching
- Photon Budget

### Tab 6: TIRF
- Penetration Depth Calculator
- Angle Sweep Mode
- Evanescent Profile Preview

---

## 📚 REFERENZEN

1. Nature Communications (2022): "Photon-free sCMOS camera characterization"
2. Biology of the Cell (2025): "Triplet state population in GFP"
3. Biophysical Journal (2019): "Calibrating TIRF penetration depths"
4. Macromolecules (2016): "Engineering elasticity in hydrogels"
5. PNAS (2013): "Axial superresolution via multiangle TIRF"
6. ScienceDirect (2025): "Piezo stage hysteresis compensation"

---

**Version:** 6.0
**Datum:** November 2025
**Status:** Implementierung in Arbeit
