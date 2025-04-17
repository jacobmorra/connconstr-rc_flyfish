import numpy as np

def rk4vec_ext(u0, t0, dt, f, args):
    k1 = dt * f ( u0, t0, *args)
    k2 = dt * f ( u0 +  k1 / 2.0, t0 + dt / 2.0, *args)
    k3 = dt * f ( u0 +  k2 / 2.0, t0 + dt / 2.0, *args)
    k4 = dt * f ( u0 +  k3, t0 + dt, *args)
    u = u0 +  (1/6)*( k1 + 2.0 * k2 + 2.0 * k3 + k4 )
    return u
