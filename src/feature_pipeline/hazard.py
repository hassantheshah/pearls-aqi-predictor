"""AQI threshold and lightweight fuzzy hazard classification."""

def _triangle(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    return (x-a)/(b-a) if x < b else (c-x)/(c-b)


def fuzzy_hazard(aqi: float) -> dict:
    """Return membership scores and the dominant fuzzy AQI risk class."""
    x = float(aqi)
    memberships = {
        "Good": max(0.0, min(1.0, (100-x)/50)) if x <= 100 else 0.0,
        "Moderate": _triangle(x, 50, 75, 120),
        "Unhealthy for Sensitive": _triangle(x, 100, 125, 175),
        "Unhealthy": _triangle(x, 150, 175, 225),
        "Very Unhealthy": _triangle(x, 200, 250, 325),
        "Hazardous": max(0.0, min(1.0, (x-300)/100)) if x >= 300 else 0.0,
    }
    # Add a plateau-like hazardous membership for values above 400.
    if x >= 400:
        memberships["Hazardous"] = 1.0
    dominant = max(memberships, key=memberships.get)
    return {"dominant": dominant, "memberships": {k: round(v, 3) for k, v in memberships.items()}}
