import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import fsolve

# If we want change the initial years:
def generate_target_years(start_year, end_year=2050, step=5, min_gap=2):
    """
    Returns a list of years from start_year to end_year (included), spaced by 'step',
    ensuring at least 'min_gap' years between start_year and the next value.

    Example: generate_target_years(2023) → [2023, 2025, 2030, 2035, 2040, 2045, 2050]
    """
    years = [start_year]
    next_year = ((start_year // step) + 1) * step
    while next_year <= end_year:
        if next_year - start_year >= min_gap:
            years.append(next_year)
        next_year += step
    return years


# LINEAR: how to use, exemple :
def linear_growth(start_year, start_value, end_year, end_value, target_years):
    """
    Computes linear interpolation between start_value and end_value over time.

    Returns a list of values for each year in target_years.

    Example: linear_growth(2020, 10, 2030, 20, [2020, 2025, 2030]) → [10.0, 15.0, 20.0]
    """
    return [round(start_value + (end_value - start_value) * (y - start_year) / (end_year - start_year), 2)
            for y in target_years]


# General LINEAR and/or Constant Function with CONTROL POINT:
def linear_with_middle_point(start_year, start_value, control_year, control_value, end_year, end_value, target_years):
    """
    Returns values with piecewise linear growth: 
    from start to control point, then from control to end.

    Useful to model changes with an intermediate value at a specific year.

    Example: control point at 2030 with a plateau or a break in slope.
    """
    values = []
    for y in target_years:
        if y <= control_year:
            if control_value == start_value:
                val = start_value
            else:
                val = start_value + (control_value - start_value) * (y - start_year) / (control_year - start_year)
        else:
            if control_value == end_value:
                val = end_value
            else:
                val = control_value + (end_value - control_value) * (y - control_year) / (end_year - control_year)
        values.append(round(val, 2))
    return values

def curved_with_middle(start_year, start_value, control_year, control_value, end_year, end_value, target_years, shape_start=1.0, shape_end=1.0, smooth_power=5):
    """
    Returns a smooth curve between three points: start, control, and end.

    The curve is made of two segments (start→control and control→end), each shaped 
    by a curvature factor:
    - shape_start and shape_end ∈ [0, 1], where 0 = linear and 1 = fully curved.

    The 'smooth_power' parameter controls the strength of curvature when shape = 1.
    Higher values produce more pronounced acceleration or deceleration near the control point.

    Useful for modeling soft transitions that are not purely linear.
    """
    def norm(x, a, b):
        return (x - a) / (b - a)

    p1 = 1 + shape_start * (smooth_power - 1)
    p2 = 1 + shape_end   * (smooth_power - 1)
    
    result = []
    for y in target_years:
        if y <= control_year:
            t = norm(y, start_year, control_year)
            v = start_value + (control_value - start_value) * (t ** p1)
        else:
            u = norm(y, control_year, end_year)
            v = control_value + (end_value - control_value) * (1 - (1 - u) ** p2)
        result.append(round(v, 2))
    return result
    

# S-CURVE that must pass through the CONTROL POINT:
def s_curve_custom(start_year, start_value, control_year, control_value, end_year, end_value, target_years, slope_factor):
    """
    Returns a smooth S-shaped curve passing through a fixed control point.

    The curve uses a logistic function centered between start and end years,
    scaled to match the given start and end values.

    If the control point is not centered, the function automatically adjusts the slope 
    (steepness) so that the curve still passes through it.

    - 'slope_factor' controls how steep the S-curve is (higher = faster transition).

    Useful for modeling progressive adoption, saturation, or demand evolution.
    """
    # Normalization of the abscissa
    def normalize(y):
        return (y - start_year) / (end_year - start_year)
    x_ctrl = normalize(control_year)
    y0, y1, y2 = start_value, control_value, end_value
    c = 0.5  # always centered in the temporal middle
    
    # Raw logistic
    def f_raw(x, k):
        return 1 / (1 + np.exp(-k * (x - c)))

    # Normalized logistic to exactly match endpoints y0 and y2
    def f_norm(x, k):
        f0 = f_raw(0, k)
        f1 = f_raw(1, k)
        return y0 + (y2 - y0) * (f_raw(x, k) - f0) / (f1 - f0)
    
    # Determine k
    mid_time = (start_year + end_year) / 2
    if control_year == mid_time:
        # Case 1: the control point is exactly in the middle
        k0 = 10.0 * slope_factor
    else:
        # Case 2: find k to pass through the control point.
        def objective(k):
            return f_norm(x_ctrl, k) - y1
        k_guess = 10.0 if y1 > (y0 + y2)/2 else -10.0 
        k0 = fsolve(objective, k_guess)[0]
    
    # Generates the curve for each target year
    x_targets = [normalize(y) for y in target_years]
    return [round(float(f_norm(x, k0)), 2) for x in x_targets]



# ACCELERATED GROWTH 
def accelerated_growth(start_year, start_value, end_year, end_value, target_years):
    """
    Returns a curved growth or decay between start and end values with an accelerating shape.

    Uses a cubic easing function to model a smooth, gradual start followed by faster change.

    Useful for scenarios like technology uptake, emissions reduction, or policy impact ramps.
    """
    t = np.array([(y - start_year) / (end_year - start_year) for y in target_years])
    vals = start_value + (end_value - start_value) * (3 - 2 * t) * t**2
    return [round(v, 2) for v in vals]

# S-CURVE (logistic function centered on midpoint)
def s_curve_growth(start_year, start_value, end_year, end_value, target_years, midpoint, sigma=0.15):
    """
    Returns an S-shaped growth curve based on a normal CDF, scaled between start and end values.

    The curve is centered on 'midpoint' and controlled by 'sigma':
    - A small sigma → steep transition (quick change)
    - A large sigma → smoother, slower transition

    The result is normalized to ensure it starts at start_value and ends at end_value.

    Useful for modeling gradual adoption, saturation effects, or phased rollouts.
    """
    if midpoint is None:
        midpoint = (start_year + end_year) / 2
    t_norm = [(y - start_year) / (end_year - start_year) for y in target_years]
    
    # Solve mu such that CDF(mu+1) = 0.9
    PROB_UMAX = 0.9
    def eq(mu, prob): return norm.cdf(1, mu, sigma) - prob
    mu = fsolve(eq, 0, args=PROB_UMAX)[0]
    
    # Apply scaled normal CDF
    s_vals = [1 - norm.cdf((t - 0.5) * 2, mu, sigma) for t in t_norm]
    s_vals = (np.array(s_vals) - min(s_vals)) / (max(s_vals) - min(s_vals))  # normalize to [0, 1]
    vals = start_value + s_vals * (end_value - start_value)
    return [round(v, 2) for v in vals]
