"""
Exercise 5 - Monte Carlo simulation of armour damage S
Van der Meer, plunging waves, no partial coefficients (Lecture 11, slide 33):
    S = ( Hs N^0.1 / (a P^b D Dn50 sqrt(cot a) s_m^0.25) )^5,   s_m = 2 pi Hs / (g Tm^2)
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 24})
g = 9.81                          # gravity [m/s2]
rng = np.random.default_rng(1)    # random generator, fixed seed for repeatable results

# ----------------------------------------------------------------- INPUT
Hs    = 2.10     # mean significant wave height [m] (Ex. 3, Hs^L)
#FIXME
Tm    = 3.95     # mean wave period [s] (Ex. 4, Tz = Tp/1.29)
#FIXME
Dn50  = 0.92     # armour nominal diameter [m] (Ex. 4, HMA 1000-3000)
#FIXME
rho_s = 2.65     # rock density [t/m3] (Ex. 4)
#FIXME
cota  = 1.5      # slope (Ex. 4)
#FIXME
P     = 0.4      # notional permeability (Ex. 4)
#FIXME
N     = 2733     # number of waves, 3*3600/Tz (Ex. 4)
#FIXME
rho_w = 1.025    # sea water density [t/m3]
a, b  = 6.2, 0.18   # van der Meer plunging: coefficient a, exponent b on P

# standard deviations [fraction of mean]
sd_Hs_list = [0.05, 0.10, 0.20]   # Hs: 5, 10, 20 %
sd_Tm, sd_rho, sd_Dn, sd_a, sd_b = 0.10, 0.05, 0.05, 0.10, 0.10   # Tm, rock density, Dn50, a, b

n = 1_000_000                 # number of simulated points
S_check = [2, 4, 6, 8]        # damage levels to report P(S > s) for

# ------------------------------------------- EXCEEDANCE CURVES
plt.figure(figsize=(12, 8))
for sd_Hs in sd_Hs_list:
    Hs_r  = rng.normal(Hs, sd_Hs*Hs, n)
    Tm_r  = rng.normal(Tm, sd_Tm*Tm, n)
    rho_r = rng.normal(rho_s, sd_rho*rho_s, n)
    Dn_r  = rng.normal(Dn50, sd_Dn*Dn50, n)
    a_r   = rng.normal(a, sd_a*a, n)
    b_r   = rng.normal(b, sd_b*b, n)

    D_r = rho_r/rho_w - 1
    sm_r = 2*np.pi*Hs_r/(g*Tm_r**2)
    S = (Hs_r*N**0.1/(a_r*P**b_r*D_r*Dn_r*np.sqrt(cota)*sm_r**0.25))**5

    S_sorted = np.sort(S)
    P_exc = 1 - np.arange(1, n + 1)/n
    plt.semilogy(S_sorted, P_exc, lw=2.5, label=f"$\\sigma_{{Hs}}$ = {sd_Hs*100:.0f} %")

    print(f"sd(Hs) = {sd_Hs*100:.0f} %:  " +
          ",  ".join(f"P(S>{s}) = {np.mean(S > s)*100:.2f} %" for s in S_check))

for s in S_check:
    plt.axvline(s, color="gray", ls=":", lw=1)
plt.xlim(0, 15)
plt.ylim(1e-4, 1)
plt.xlabel("Damage S [-]")
plt.ylabel("Probability of exceedance [-]")
plt.title("Monte Carlo simulation of armour damage")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("mc_exceedance.png", dpi=150)

# --------------------------------------- SENSITIVITY TO n (sd(Hs) = 10 %)
n_list = np.logspace(2, 6, 25).astype(int)   # n from 1e2 to 1e6, 25 log-spaced values
P_n = np.zeros((len(n_list), len(S_check)))
for i, ni in enumerate(n_list):
    Hs_r  = rng.normal(Hs, 0.10*Hs, ni)
    Tm_r  = rng.normal(Tm, sd_Tm*Tm, ni)
    rho_r = rng.normal(rho_s, sd_rho*rho_s, ni)
    Dn_r  = rng.normal(Dn50, sd_Dn*Dn50, ni)
    a_r   = rng.normal(a, sd_a*a, ni)
    b_r   = rng.normal(b, sd_b*b, ni)
    D_r = rho_r/rho_w - 1
    sm_r = 2*np.pi*Hs_r/(g*Tm_r**2)
    S = (Hs_r*N**0.1/(a_r*P**b_r*D_r*Dn_r*np.sqrt(cota)*sm_r**0.25))**5
    P_n[i] = [np.mean(S > s) for s in S_check]

plt.figure(figsize=(12, 8))
for j, s in enumerate(S_check):
    plt.semilogx(n_list, P_n[:, j]*100, "o-", lw=2, label=f"S > {s}")
plt.xlabel("Number of simulated points n [-]")
plt.ylabel("Probability of exceedance [%]")
plt.title("Sensitivity to n ($\\sigma_{Hs}$ = 10 %)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("mc_sensitivity_n.png", dpi=150)
plt.show()
