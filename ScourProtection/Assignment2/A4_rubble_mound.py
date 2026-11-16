import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import gamma, gammainc, gammaincc

g = 9.81

# ------------------------------------------------------------------ Results from Exercise 3
# GPD: H(T) = u + sigma/xi*((lam*T)**xi - 1)
u      = 1.0     # POT threshold [m]
#FIXME
xi     = -0.134  # GPD shape
#FIXME
sigma  = 0.259   # GPD scale [m]
#FIXME
lam    = 10.9    # POT events per year [1/yr]
#FIXME
Ndata  = 109     # number of POT events used in the fit
#FIXME
aTp, bTp = 4.08, 0.300   # Tp = aTp * Hs**bTp
#FIXME

# ----------------------------------------------------------------- INPUT
L     = 50       # lifetime [yr]
pf    = 0.2      # probability of failure in lifetime
sigF  = 0.1      # sigma'_FHs, PIANC Table 4.1 (0.05-0.1 buoy, 0.1-0.2 hindcast)
ks    = 0.05

d     = 7.0      # water depth rel. MSL [m]
HW, LW = 1.2, -1.0
tanb  = 0.01     # foreshore slope
cota  = 1.5      # front & rear slope
S     = 2.0
P     = 0.4
rho_s, rho_w = 2.65, 1.025
D = rho_s/rho_w - 1

D15c, D50c, D85c = 0.025, 0.080, 0.250   # core [m]

z_rf, z_rc = -2.0, -4.0     # rear filter / core berm level
W_fb_front = 3              # front filter berm width [armour Dn50]
W_cb, W_rf, W_rc = 2.0, 2.0, 2.0   # [m]

# EN 13383 (Rock Manual Tables 3.5 & 3.6): name, M50min, M50max [kg]
EN13383 = [("LMA 5-40", 14, 28), ("LMA 10-60", 27, 47), ("LMA 40-200", 101, 152),
           ("LMA 60-300", 162, 236), ("HMA 300-1000", 628, 802),
           ("HMA 1000-3000", 1870, 2310), ("HMA 3000-6000", 4430, 5060),
           ("HMA 6000-10000", 7680, 8710), ("HMA 10000-15000", 12000, 13000)]
# coarse gradings: name, D50 = 0.5(NLL+NUL) [m]  (Table 3.6)
CP = [("CP 45/125", 0.085), ("CP 63/180", 0.122), ("CP 90/250", 0.170)]

# ------------------------------------------------- DESIGN WAVE HEIGHTS
Tpf = 1/(1 - (1 - pf)**(1/L))
HsL   = u + sigma/xi*((lam*L)**xi - 1)
Hs3L  = u + sigma/xi*((lam*3*L)**xi - 1)
HsTpf = u + sigma/xi*((lam*Tpf)**xi - 1)
Tp = aTp*HsL**bTp
Tz = Tp/1.29
Nz = 3*3600/Tz
print(f"\nTpf = {Tpf:.0f} yr,  Hs^L = {HsL:.2f} m,  Hs^3L = {Hs3L:.2f} m,  Hs^Tpf = {HsTpf:.2f} m")
print(f"Tp = {Tp:.2f} s,  Tz = {Tz:.2f} s,  N = {Nz:.0f}")

# --------------------------------- H2% at HW (BG2000)
h = d + HW
Hrms = (2.69 + 3.24*(HsL/4)/h)*HsL/4          # Hm0 = 4 sqrt(m0)
Htr = (0.35 + 5.8*tanb)*h/Hrms                 # normalised by Hrms
k1, k2 = 2.0, 3.6
H2n = lambda H1: Htr*(H1/Htr)**(k1/k2)
res = lambda H1: np.sqrt(H1**2*gammainc(2/k1+1, (Htr/H1)**k1)*gamma(2/k1+1)
                         + H2n(H1)**2*gammaincc(2/k2+1, (Htr/H2n(H1))**k2)*gamma(2/k2+1)) - 1
H1 = brentq(res, 0.5, 50) if Htr < 3 else 1.0
H2pct = H1*np.sqrt(-np.log(0.02))
if H2pct > Htr:
    H2pct = H2n(H1)*(-np.log(0.02))**(1/k2)
H2pct *= Hrms
print(f"h = {h:.1f} m, h/Hs = {h/HsL:.1f},  Htr/Hrms = {Htr:.2f},  H2% = {H2pct:.2f} m  (H2%/Hs = {H2pct/HsL:.2f})")

# --------------------------------------------------------- MAIN ARMOUR
# H2% form of van der Meer (lecture slide 29); equals the deep-water form when Rayleigh holds
sm = 2*np.pi*HsL/(g*Tz**2)
xim = 1/(cota*np.sqrt(sm))
ximc = (6.2*P**0.31*np.sqrt(1/cota))**(1/(P + 0.5))
if xim < ximc:
    breaker, ka, kb = "plunging", 0.027, 38       # PIANC Table 7.3
    Ns = 8.7*S**0.2*P**0.18*Nz**-0.1*xim**-0.5
else:
    breaker, ka, kb = "surging", 0.031, 38
    Ns = 1.4*S**0.2*P**-0.13*Nz**-0.1*np.sqrt(cota)*xim**P

gz = 1 - ka*np.log(pf)
gH = HsTpf/HsL + sigF**(1 + (Hs3L/HsL - 1)*kb*pf) + ks/np.sqrt(pf*Ndata)
Dn50a = gz*gH*H2pct/(D*Ns)
print(f"\nxi_m = {xim:.2f}, xi_mc = {ximc:.2f} -> {breaker}")
print(f"gamma_z = {gz:.3f}, gamma_Hs = {gH:.3f}, H2%/(D Dn50) = {Ns:.2f}")
print(f"Dn50 (conventional crest) = {Dn50a:.2f} m")

# Low crest (PIANC Table 7.4), Rc/Hs = 0.6
Rc = 0.6*HsL
ka, kb = 0.035, 42
gz_lc = 1 - ka*np.log(pf)
gH_lc = HsTpf/HsL + sigF**(1 + (Hs3L/HsL - 1)*kb*pf) + ks/np.sqrt(pf*Ndata)
sop = 2*np.pi*HsL/(g*Tp**2)
fi = min(1/(1.25 - 4.8*Rc/HsL*np.sqrt(sop/(2*np.pi))), 1.0)   # no reduction outside validity
Dn50_req = fi*Dn50a*gz_lc*gH_lc/(gz*gH)
M50_req = rho_s*1000*Dn50_req**3
print(f"Low crest: fi = {fi:.3f}, gamma_z = {gz_lc:.3f}, gamma_Hs = {gH_lc:.3f}")
print(f"Required armour: Dn50 = {Dn50_req:.2f} m, M50 = {M50_req:.0f} kg")

# EN 13383: smallest grading with M50min >= required M50
arm = next(gr for gr in EN13383 if gr[1] >= M50_req)
M50a = 0.5*(arm[1] + arm[2])
Dn50a = (M50a/(rho_s*1000))**(1/3)
print(f"Armour grading {arm[0]}: M50 = {M50a:.0f} kg, Dn50 = {Dn50a:.2f} m")

z_crest = HW + Rc
ta = 2*Dn50a
Bc = 4*Dn50a
print(f"Rc = {Rc:.2f} m -> crest at +{z_crest:.2f} m,  t_armour = {ta:.2f} m,  B = {Bc:.2f} m")

# ------------------------------------------------------- FILTER LAYER
D50a = Dn50a/0.84
d15a, d85a = D50a/1.3, D50a*1.3
D85f_min, D50f_min, D15f_min = d15a/4, D50a/7, d15a/7     # Thompson & Shuttler
print(f"\nArmour D15/D50/D85 = {d15a:.2f}/{D50a:.2f}/{d85a:.2f} m")
print(f"Filter minimum D15/D50/D85 = {D15f_min:.3f}/{D50f_min:.3f}/{D85f_min:.3f} m")

# candidates sorted by D50 (sieve size): coarse + light/heavy gradings
cand = CP + [(n, ((m1 + m2)/2/(rho_s*1000))**(1/3)/0.84) for n, m1, m2 in EN13383]
fil, D50f = next(c for c in sorted(cand, key=lambda c: c[1])
                 if c[1]/1.3 >= D15f_min and c[1] >= D50f_min and c[1]*1.3 >= D85f_min)
D15f, D85f = D50f/1.3, D50f*1.3
Dn50f = 0.84*D50f
print(f"Filter grading {fil}: D15/D50/D85 = {D15f:.3f}/{D50f:.3f}/{D85f:.3f} m, "
      f"M50 = {rho_s*1000*Dn50f**3:.0f} kg")

c1, c2, c3 = D15f/D85c, D50f/D50c, D15f/D15c
ok = c1 <= 4 and c2 <= 7 and c3 <= 7
print(f"Filter/core: D15f/D85c = {c1:.2f} (<=4), D50f/D50c = {c2:.2f} (<=7), "
      f"D15f/D15c = {c3:.2f} (<=7) -> {'OK' if ok else 'NOT OK'}")
tf = max(0.5, 2*D50f)
print(f"t_filter = {tf:.2f} m")

# ---------------------------------------------------------- TOE BERMS
# PIANC Table 7.5: (1/gz) 8.7 (ht/hs)^1.43 D Dn50 >= gHs Hs  ->  solve ht (valid ht/hs > 0.5)
hs = d + LW
ka, kb = 0.087, 100
gz_t = 1 - ka*np.log(pf)
gH_t = HsTpf/HsL + sigF**(1 + (Hs3L/HsL - 1)*kb*pf) + ks/np.sqrt(pf*Ndata)
print(f"\nToe (LW, hs = {hs:.1f} m): gamma_z = {gz_t:.3f}, gamma_Hs = {gH_t:.3f}")
# start with filter stones; go to larger gradings until berm (>= 2 stones thick) is stable above seabed
for toe, D50t in sorted(cand, key=lambda c: c[1]):
    Dn50t = 0.84*D50t
    ht_f = hs*(gz_t*gH_t*HsL/(8.7*D*Dn50t))**(1/1.43)
    print(f"  {toe:16s} Dn50 = {Dn50t:.3f} m: ht = {ht_f:.2f} m (ht/hs = {ht_f/hs:.2f}) -> top at {LW - ht_f:.2f} m")
    if D50t >= D50f and LW - ht_f >= -d + 2*Dn50t:
        break
z_bf = LW - ht_f
ht_c = hs*(gz_t*gH_t*HsL/(8.7*D*0.84*D50c))**(1/1.43)
z_bc = LW - ht_c
print(f"Toe berm: {toe}, top at {z_bf:.2f} m")
print(f"Core berm: ht = {ht_c:.2f} m -> top at {z_bc:.2f} m "
      f"{'(below seabed: no core berm)' if z_bc <= -d else ''}")
# cross-check, Rock Manual eq. 5.188 with Nod = 0.5 (no partial coefficients)
Dn_588 = HsL/(D*(2 + 6.2*(ht_f/hs)**2.7)*0.5**0.15)
print(f"Check RM eq. 5.188 (Nod = 0.5): required Dn50 = {Dn_588:.3f} m vs toe Dn50 = {Dn50t:.3f} m")

# ------------------------------------------------------------ DRAWING
a, zs, r = cota, -d, np.sqrt(1 + cota**2)
xf = lambda z, off: -Bc/2 - a*(z_crest - z) + off*r     # front slope, normal offset 'off'
xr = lambda z, off: Bc/2 + a*(z_crest - z) - off*r      # rear slope

x_fb = xf(z_bf, 0) - W_fb_front*Dn50a                  # front filter berm edge
core_berm = z_bc > zs
z_low = z_bc if core_berm else zs
x_ff = x_fb - a*(z_bf - z_low)
front = [(x_ff - W_cb - a*(z_bc - zs), zs), (x_ff - W_cb, z_bc), (x_ff, z_bc)] if core_berm else [(x_ff, zs)]
front += [(x_fb, z_bf)]
x_rfe = xr(z_rf, 0) + W_rf
x_rce = x_rfe + a*(z_rf - z_rc) + W_rc
rear = [(x_rfe, z_rf), (x_rfe + a*(z_rf - z_rc), z_rc), (x_rce, z_rc), (x_rce + a*(z_rc - zs), zs)]

S_out = front + [(xf(z_bf, 0), z_bf), (xf(z_crest, 0), z_crest), (xr(z_crest, 0), z_crest), (xr(z_rf, 0), z_rf)] + rear
F_out = front + [(xf(z_bf, ta), z_bf), (xf(z_crest - ta, ta), z_crest - ta),
                 (xr(z_crest - ta, ta), z_crest - ta), (xr(z_rf, ta), z_rf)] + rear
t2 = ta + tf
C_front = front[:3] + [(xf(z_bc, t2), z_bc)] if core_berm else [(xf(zs, t2), zs)]
C_out = C_front + [(xf(z_crest - t2, t2), z_crest - t2), (xr(z_crest - t2, t2), z_crest - t2),
                   (xr(z_rc, t2), z_rc)] + rear[2:]

fig, ax = plt.subplots(figsize=(14, 5))
for poly, col, lab in [(S_out, "dimgray", f"Armour {arm[0]} (Dn50 = {Dn50a:.2f} m)"),
                       (F_out, "darkgray", f"Filter {fil} (t = {tf:.2f} m), toe {toe}"),
                       (C_out, "wheat", "Core 25-250 mm")]:
    ax.fill(*zip(*poly), color=col, ec="k", lw=0.8, label=lab)
xmin, xmax = S_out[0][0] - 3, S_out[-1][0] + 3
for z, ls, lab in [(HW, "--", "HW +1.2"), (0, "-", "MSL"), (LW, ":", "LW -1.0")]:
    ax.plot([xmin, xmax], [z, z], "b", ls=ls, lw=0.8)
    ax.text(xmin, z + 0.1, lab, color="b", fontsize=8)
ax.plot([xmin, xmax], [zs, zs], "k", lw=1.5)
for z in [z_crest, z_bf, z_rf, z_rc] + ([z_bc] if core_berm else []):
    ax.text(xmax, z, f"{z:+.2f}", fontsize=8, va="center")
ax.set_aspect("equal")
ax.set_xlim(xmin, xmax + 2)
ax.set_ylim(zs - 1, z_crest + 3)
ax.set_xlabel("x [m]  (sea side left)")
ax.set_ylabel("Level rel. MSL [m]")
ax.set_title(f"Detached rubble mound breakwater, cot a = {cota}, B = {Bc:.2f} m")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("breakwater_section.png", dpi=150)
plt.show()
