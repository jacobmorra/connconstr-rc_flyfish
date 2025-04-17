import numpy as np

def rk4vec(u0, t0, dt, f):
    k1 = dt * f ( u0, t0 )
    k2 = dt * f ( u0 +  k1 / 2.0, t0 + dt / 2.0)
    k3 = dt * f ( u0 +  k2 / 2.0, t0 + dt / 2.0 )
    k4 = dt * f ( u0 +  k3, t0 + dt )
    u = u0 +  (1/6)*( k1 + 2.0 * k2 + 2.0 * k3 + k4 )
    return u
