import math
from datetime import date


def compute_solar_declination(d):
    n_0 = 78.801 + 0.2422 * (d.year - 1969) - int(0.25 * (d.year - 1969))
    n = (d - date(d.year, 1, 1)).days
    w = 360 * (n - n_0 - 0.5) / 365.2422
    w = math.radians(w)
    delta = (
        0.3723
        + 23.2567 * math.sin(w)
        + 0.1149 * math.sin(2 * w)
        - 0.1712 * math.sin(3 * w)
        - 0.7580 * math.cos(w)
        + 0.3656 * math.cos(2 * w)
        + 0.0201 * math.cos(3 * w)
    )
    return delta


def get_solarhourangle(alpha, phi, delta, isMorning=True):
    alpha, phi, delta = math.radians(alpha), math.radians(phi), math.radians(delta)
    a = (math.tan(alpha) ** 2) * (math.sin(phi) ** 2) + 1
    b = -math.sin(2 * phi) * math.tan(delta) * (math.tan(alpha) ** 2)
    c = (math.tan(alpha) ** 2) * (math.cos(phi) ** 2) * (math.tan(delta) ** 2) - 1

    gamma = min(
        math.acos((-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)),
        math.acos((-b - math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)),
    )

    if not isMorning:
        gamma *= -1

    return math.degrees(gamma)


def get_La1a2(a2b, a1b, alpha_s, alpha_sa):
    alpha_diff = math.radians(alpha_s - alpha_sa)
    a1a2_square = (a2b ** 2) + (a1b ** 2) - (2 * a1b * a2b * math.cos(alpha_diff))
    return math.sqrt(a1a2_square)


def get_solarelevation(phi, delta, omega):
    phi, delta, omega = math.radians(phi), math.radians(delta), math.radians(omega)
    h_s = math.asin(
        math.sin(phi) * math.sin(delta)
        + math.cos(phi) * math.cos(delta) * math.cos(omega)
    )
    return math.degrees(h_s)


def get_ratios(hs, alpha_s, alpha_sa, la2b, la1a2):
    hs, alpha_diff = math.radians(hs), math.radians(alpha_s - alpha_sa)

    H = math.tan(hs) * (
        la2b * math.cos(alpha_diff)
        + math.sqrt((la1a2 ** 2) - (la1a2 ** 2) * (math.sin(alpha_diff) ** 2))
    )
    return H / la2b, H / la1a2


def estimate_height(length, ratio):
    return length * ratio
