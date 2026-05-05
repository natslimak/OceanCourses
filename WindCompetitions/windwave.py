import math


def wind_pressure(rho_air: float, Cd: float, V: float) -> float:
	"""Dynamic wind pressure (N/m^2)."""
	return 0.5 * rho_air * Cd * V * V


def wind_force_disc(p: float, R: float) -> float:
	"""Wind force on a circular disc of radius R (m)."""
	A = math.pi * R * R
	return p * A


def tower_wind_drag_per_length(rho_air: float, Cd: float, D: float, V: float) -> float:
	"""Drag per unit length on a vertical cylinder (N/m)."""
	return 0.5 * rho_air * Cd * D * V * V


def orbital_velocity_amp(H: float, T: float) -> float:
	"""Approximate orbital velocity amplitude at the surface: U = pi*H/T"""
	return math.pi * H / T


def orbital_accel_amp(U: float, T: float) -> float:
	"""Acceleration amplitude from velocity amplitude: a = (2*pi/T)*U"""
	omega = 2.0 * math.pi / T
	return omega * U


def morison_per_length(rho: float, Cd: float, Cm: float, D: float, U: float, a: float) -> tuple:
	"""Return (drag_per_length, inertia_per_length, total_per_length).

	drag: 0.5*rho*Cd*D*U*|U|
	inertia: rho*Cm*(pi*D^2/4)*a
	"""
	f_d = 0.5 * rho * Cd * D * U * abs(U)
	f_i = rho * Cm * (math.pi * D * D / 4.0) * a
	return f_d, f_i, f_d + f_i


def equivalent_point_loads(hub_height: float = None) -> None:
	"""Compute and print equivalent point loads and moments for a single turbine/column.

	If `hub_height` is None the global `HUB_HEIGHT` is used.
	"""
	if hub_height is None:
		hub_height = HUB_HEIGHT
	# wave kinematics
	U = orbital_velocity_amp(Hs, Tp)
	a = orbital_accel_amp(U, Tp)
	_, _, f_tot = morison_per_length(rho, Cd, Cm, D, U, a)
	F_wave = f_tot * L

	# wind
	p_w = wind_pressure(rho_air, Cd_wind, V)
	F_wind = wind_force_disc(p_w, R)

	z_wave = L / 2.0  # approximate application point under top
	M_wind = F_wind * hub_height
	M_wave = F_wave * z_wave

	print()
	print("Equivalent point loads and moments (simple estimates):")
	print(" - Wind point load at hub: {:.1f} N (applied at hub height {:.2f} m)".format(F_wind, hub_height))
	print(" - Equivalent wind moment about top: {:.1f} N·m".format(M_wind))
	print(" - Wave equivalent point load on column: {:.1f} N (applied at {:.2f} m below top)".format(F_wave, z_wave))
	print(" - Equivalent wave moment about top: {:.1f} N·m".format(M_wave))



# Editable parameters: change these values directly for quick estimates
# Wind
V = 10.0            # wind speed (m/s)
Cd_wind = 1.2       # wind drag coeff for bluff body
rho_air = 1.225     # air density (kg/m3)

# Waves / Morison
Hs = 0.48           # significant wave height Hs (m)
Tp = 4.3            # peak period Tp (s)
Cd = 1.0            # wave/drag coefficient (dimensionless)
Cm = 2.0            # added mass coefficient (dimensionless)
rho = 1025.0        # water density (kg/m3)

# Geometry
D = 0.315           # cylinder diameter D (m)
L = 0.3             # submerged length L (m)
R = 0.6            # rotor radius R (m) for wind disc

# Structural
HUB_HEIGHT = 1.5    # hub height above top connection (m) — edit as needed




def main():
	# Use module-level editable variables defined near the top of this file
	# Wind
	p_wind = wind_pressure(rho_air, Cd_wind, V)
	F_wind_disc = wind_force_disc(p_wind, R)
	f_tower_wind = tower_wind_drag_per_length(rho_air, Cd_wind, D, V)

	# Waves (use Hs as H)
	H = Hs
	T = Tp
	U = orbital_velocity_amp(H, T)
	a = orbital_accel_amp(U, T)
	f_d, f_i, f_tot = morison_per_length(rho, Cd, Cm, D, U, a)
	F_wave_total = f_tot * L

	# Print results
	print("Quick wind and Morison wave load estimates")
	print("Inputs: V={:.2f} m/s, Hs={:.2f} m, Tp={:.2f} s, D={:.3f} m, L={:.2f} m".format(
		V, Hs, Tp, D, L))
	print()
	print("Wind:")
	print(" - wind dynamic pressure p = {:.2f} N/m^2".format(p_wind))
	print(" - rotor/disc force (R={:.2f} m): F = {:.1f} N".format(R, F_wind_disc))
	print(" - tower drag per length = {:.2f} N/m".format(f_tower_wind))
	print()
	print("Waves (Morrison per unit length):")
	print(" - orbital velocity amp U = {:.3f} m/s".format(U))
	print(" - orbital accel amp a = {:.3f} m/s^2".format(a))
	print(" - drag per length = {:.2f} N/m".format(f_d))
	print(" - inertia per length = {:.2f} N/m".format(f_i))
	print(" - total per length = {:.2f} N/m".format(f_tot))
	print(" - total wave horizontal force on cylinder (L={:.2f} m) = {:.1f} N".format(L, F_wave_total))

	# Equivalent point loads & moments
	equivalent_point_loads(HUB_HEIGHT)


if __name__ == '__main__':
	main()

