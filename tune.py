"""
Oscillation tracking for controller tuning and analysis.

Tracks peak-to-peak amplitude and oscillation period for each degree of freedom
to help diagnose controller performance and detect limit cycles.
"""


class DOFOscillationTracker:
    """Tracks oscillation characteristics across multiple degrees of freedom."""
    
    def __init__(self):
        self.data = {}

    def _init_axis(self, name, initial_value, t):
        """Initializes tracking data structure for a new axis."""
        self.data[name] = {
            "prev": initial_value,
            "prev_slope": 0,
            "max": initial_value,
            "min": initial_value,
            "t_max": t,
            "t_min": t,
            "last_peak_time": None,
            "period": 0.0,
            "last_local_max": initial_value,
            "last_local_min": initial_value,
            "current_amplitude": 0.0,
        }

    def update(self, name, value, t):
        """Updates oscillation metrics for one DOF using peak detection."""
        if name not in self.data:
            self._init_axis(name, value, t)

        d = self.data[name]

        if value > d["max"]:
            d["max"] = value
            d["t_max"] = t

        if value < d["min"]:
            d["min"] = value
            d["t_min"] = t

        slope = value - d["prev"]
        sign = 1 if slope > 0 else -1

        if sign != d["prev_slope"] and d["prev_slope"] != 0:
            if d["prev_slope"] > 0 and sign < 0:
                d["last_local_max"] = value

                d["current_amplitude"] = abs(d["last_local_max"] - d["last_local_min"])

                if d["last_peak_time"] is None:
                    d["last_peak_time"] = t
                else:
                    d["period"] = t - d["last_peak_time"]
                    d["last_peak_time"] = t

            elif d["prev_slope"] < 0 and sign > 0:
                d["last_local_min"] = value

                d["current_amplitude"] = abs(d["last_local_max"] - d["last_local_min"])

        d["prev"] = value
        d["prev_slope"] = sign

    def get_info(self, name):
        """Returns current tracking statistics for the specified DOF."""
        d = self.data[name]
        return {
            "max": d["max"],
            "min": d["min"],
            "t_max": d["t_max"],
            "t_min": d["t_min"],
            "period": d["period"],
            "current_amplitude": d["current_amplitude"],
        }
