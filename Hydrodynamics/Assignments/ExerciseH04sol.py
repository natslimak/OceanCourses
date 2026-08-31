import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import fsolve


# %% General Parameters
g = 9.81         # (m/s^2) Gravitational Acceleration

nu = 1e-6        # (m^2/s) Kinematic Viscosity of Water

# %% Question 1
h = 10.0   # (m) Water depth
H = 0.5    # (m) Wave height
T = np.array([3, 7, 22])  # (s) Wave period

omega = 2 * np.pi / T        # (1/s) Wave angular frequency
k0 = omega**2 / g            # Initial guess for wavenumber
k = np.zeros_like(omega)

# Solve dispersion relation
for i in range(len(omega)):
    func = lambda k_: omega[i]**2 - g * k_ * np.tanh(k_ * h)
    k[i] = fsolve(func, k0[i])[0]

L = 2 * np.pi / k      # Wavelength
kh = h * k             # Dimensionless depth

# %% Printing results
for i in range(len(kh)):
    if kh[i] > np.pi:
        print(f"kh={kh[i]:.3f}, T={T[i]:.1f} s, L={L[i]:.3f} m --> Deep water")
    elif kh[i] < np.pi/10:
        print(f"kh={kh[i]:.3f}, T={T[i]:.1f} s, L={L[i]:.3f} m --> Shallow water")
    else:
        print(f"kh={kh[i]:.3f}, T={T[i]:.1f} s, L={L[i]:.3f} m --> Intermediate water")

# %% Question 2 - Animation for Deep-water case
x = np.arange(0, L[0], 0.1)
t = np.arange(0, T[0], 0.1)

eta = np.zeros((len(t), len(x)))
for i in range(len(t)):
    eta[i, :] = H/2 * np.sin(omega[0]*t[i] - k[0]*x)

plt.figure(1)
for i in range(len(t)):
    plt.clf()
    plt.plot(x, eta[i, :])
    plt.grid(True, which="both")
    plt.xlabel("x (m)")
    plt.ylabel("η (m)")
    plt.title(f"t={t[i]:.1f} s")
    plt.pause(0.01)

input("\nPress Enter to continue...\n")

# %% Question 3 - Velocity profiles
x0 = 0
y = np.arange(0, h+0.1, 0.1)
u = []
v = []
t_list = []

for i in range(len(T)):
    ti = np.arange(0, T[i], 0.1)
    t_list.append(ti)
    ui = np.zeros((len(ti), len(y)))
    vi = np.zeros((len(ti), len(y)))
    for j in range(len(ti)):
        ui[j, :] = (H/2*omega[i]*np.cosh(k[i]*y)/np.sinh(k[i]*h) *
                    np.sin(omega[i]*ti[j] - k[i]*x0))
        vi[j, :] = (H/2*omega[i]*np.sinh(k[i]*y)/np.sinh(k[i]*h) *
                    np.cos(omega[i]*ti[j] - k[i]*x0))
    u.append(ui)
    v.append(vi)

plt.figure(2)
for idx in range(3):
    plt.subplot(1, 3, idx+1)
    for j in range(len(t_list[idx])):
        plt.cla()
        plt.plot(u[idx][j, :], y, label="u (m/s)")
        plt.plot(v[idx][j, :], y, label="v (m/s)")
        plt.grid(True, which="both")
        plt.ylabel("y (m)")
        plt.legend()
        plt.title(f"t={t_list[idx][j]:.1f} s (kh={kh[idx]:.3f})")
        plt.xlim([-np.max(np.abs(u[0])), np.max(np.abs(u[0]))])
        plt.pause(0.01)

input("\nPress Enter to continue...\n")

# %% Question 4 - Particle paths
x_paths = []
y_paths = []
x0, y0 = 0, h

for i in range(len(T)):
    xi = []
    yi = []
    for j in range(len(t_list[i])):
        xi.append(x0 - H/2*np.cosh(k[i]*y0)/np.sinh(k[i]*h) *
                  np.cos(omega[i]*t_list[i][j] - k[i]*x0))
        yi.append(y0 + H/2*np.sinh(k[i]*y0)/np.sinh(k[i]*h) *
                  np.sin(omega[i]*t_list[i][j] - k[i]*x0))
    x_paths.append(xi)
    y_paths.append(yi)

plt.figure(3)
for idx in range(3):
    plt.subplot(3, 1, idx+1)
    for j in range(len(t_list[idx])):
        plt.scatter(x_paths[idx][j], y_paths[idx][j], c='b', s=10)
        plt.grid(True, which="both")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(f"t={t_list[idx][j]:.1f} s (kh={kh[idx]:.3f})")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.pause(0.01)

input("\nPress Enter to continue...\n")

# %% Question 5_1 - Deep water wave group
k1, omega1 = k[0], omega[0]
k2 = 1.1 * k1
omega2 = np.sqrt(g*k2*np.tanh(k2*h))

Tg = 4*np.pi/(omega2-omega1)
Lg = 4*np.pi/(k2-k1)

c1 = omega1/k1
c2 = omega2/k2
cg = Lg/Tg

print(f"Deep water: c1={c1:.3f}, c2={c2:.3f}, cg={cg:.3f}")

t = np.linspace(0, Tg, 100)
x = np.arange(0, Lg, 0.1)
eta_group = np.zeros((len(t), len(x)))
for i in range(len(t)):
    eta_group[i, :] = (H/2*np.sin(omega1*t[i]-k1*x) +
                       H/2*np.sin(omega2*t[i]-k2*x))

plt.figure(4)
for i in range(len(t)):
    plt.clf()
    plt.plot(x, eta_group[i, :])
    plt.grid(True, which="both")
    plt.xlabel("x (m)")
    plt.ylabel("η_g (m)")
    plt.title(f"t={t[i]:.1f} s")
    plt.pause(0.1)

input("\nPress Enter to continue...\n")

# %% Question 5_2 - Shallow water wave group
k1, omega1 = k[2], omega[2]
k2 = 1.1 * k1
omega2 = np.sqrt(g*k2*np.tanh(k2*h))

Tg = 4*np.pi/(omega2-omega1)
Lg = 4*np.pi/(k2-k1)

c1 = omega1/k1
c2 = omega2/k2
cg = Lg/Tg

print(f"Shallow water: c1={c1:.3f}, c2={c2:.3f}, cg={cg:.3f}")

t = np.linspace(0, Tg, 100)
x = np.arange(0, Lg, 0.1)
eta_group = np.zeros((len(t), len(x)))
for i in range(len(t)):
    eta_group[i, :] = (H/2*np.sin(omega1*t[i]-k1*x) +
                       H/2*np.sin(omega2*t[i]-k2*x))

plt.figure(5)
for i in range(len(t)):
    plt.clf()
    plt.plot(x, eta_group[i, :])
    plt.grid(True, which="both")
    plt.xlabel("x (m)")
    plt.ylabel("η_g (m)")
    plt.title(f"t={t[i]:.1f} s")
    plt.pause(0.1)

plt.show()

