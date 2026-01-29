class SmartDevice:
    def __init__(self, name):
        self.name = name
        self.is_broken = False
        self.status_msg = "Normal"

    def is_busy(self):
        """Returns True if device is in a time-sensitive task (cooking, washing)."""
        return False

    def tick(self, hours):
        """Advance internal timer if applicable."""
        pass

    def break_device(self):
        self.is_broken = True
        self.status_msg = "Malfunction detected"

    def repair(self):
        self.is_broken = False
        self.status_msg = "Normal"

    def read(self):
        if self.is_broken:
            return {
                "name": self.name, 
                "error": "ConnectionError", 
                "message": "Device unreachable. Please attempt repair/reboot."
            }
        
        return {"name": self.name}

    def update(self, params):
        if self.is_broken:
            return False, "Device unreachable. Please attempt repair/reboot."
        return True, "Updated"