class DOFOscillationTracker:
    def __init__(self):
        # Stores tracking data for each DOF name: "N", "E", "D", "Roll", "Pitch", "Yaw"
        self.data = {}

    def _init_axis(self, name, initial_value, t):
        self.data[name] = {
            "prev": initial_value,
            "prev_slope": 0,

            # Global extreme tracking
            "max": initial_value,
            "min": initial_value,
            "t_max": t,
            "t_min": t,

            # Oscillation timing
            "last_peak_time": None,
            "period": 0.0,

            # For amplitude tracking
            "last_local_max": initial_value,
            "last_local_min": initial_value,
            "current_amplitude": 0.0,
        }

    def update(self, name, value, t):
        """Update tracking for one DOF"""
        if name not in self.data:
            self._init_axis(name, value, t)

        d = self.data[name]

        # --- Track global max/min (not amplitude) ---
        if value > d["max"]:
            d["max"] = value
            d["t_max"] = t

        if value < d["min"]:
            d["min"] = value
            d["t_min"] = t

        # --- Derivative sign for peak detection ---
        slope = value - d["prev"]
        sign = 1 if slope > 0 else -1

        # ---------------------------------------------
        # Detect zero-crossings of slope → peaks/troughs
        # ---------------------------------------------
        if sign != d["prev_slope"] and d["prev_slope"] != 0:

            # Case 1: going + → -  (local maximum)
            if d["prev_slope"] > 0 and sign < 0:
                # Local maximum at value
                d["last_local_max"] = value

                # Compute amplitude using last trough
                d["current_amplitude"] = abs(
                    d["last_local_max"] - d["last_local_min"]
                )

                # Update period timing
                if d["last_peak_time"] is None:
                    d["last_peak_time"] = t
                else:
                    d["period"] = t - d["last_peak_time"]
                    d["last_peak_time"] = t

            # Case 2: going - → +  (local minimum)
            elif d["prev_slope"] < 0 and sign > 0:
                # Local minimum at value
                d["last_local_min"] = value

                # Compute amplitude using last peak
                d["current_amplitude"] = abs(
                    d["last_local_max"] - d["last_local_min"]
                )

        # Store current value and slope
        d["prev"] = value
        d["prev_slope"] = sign

    def get_info(self, name):
        """Return current tracking info for printing"""
        d = self.data[name]
        return {
            "max": d["max"],
            "min": d["min"],
            "t_max": d["t_max"],
            "t_min": d["t_min"],
            "period": d["period"],
            "current_amplitude": d["current_amplitude"],
        }
