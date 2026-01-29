from ..base import SmartDevice


class AirConditioner(SmartDevice):
    def __init__(self, name):
        super().__init__(name)
        self.power = "off"
        self.mode = "cooling"
        self.fan_speed = "auto"
        self.sleep_mode = "off"
        self.temperature = 26
        self.timer = 0.0  # 剩余定时关机时间 (0表示不启用定时)

    def tick(self, hours):
        """
        处理时间流逝：
        只有当空调开启且设置了定时器时，时间才会消耗定时器。
        """
        if self.power == "on" and self.timer > 0:
            if self.timer > hours:
                self.timer -= hours
            else:
                # 时间到了，定时结束，自动关机
                self.timer = 0
                self.power = "off"

    def read(self):
        base = super().read()
        base.update({
            "power": self.power, 
            "mode": self.mode, 
            "fan_speed": self.fan_speed, 
            "sleep_mode": self.sleep_mode,
            "temperature": self.temperature,
            "timer_remaining": self.timer
        })
        return base

    def update(self, params):
        success, msg = super().update(params)
        if not success: return success, msg

        changes = []
        should_turn_on = False

        # --- 1. 常规设置 ---
        if "power" in params:
            if params["power"] in ["on", "off"]:
                self.power = params["power"]
                changes.append(f"turned {self.power}")
                # 手动关机则取消所有定时任务
                if self.power == "off":
                    self.timer = 0
            else:
                return False, "Invalid power state (on/off)."

        if "mode" in params:
            if params["mode"] in ["cooling", "heating", "dehumidify"]:
                self.mode = params["mode"]
                changes.append(f"mode set to {self.mode}")
                should_turn_on = True
            else:
                return False, "Invalid mode."

        if "fan_speed" in params:
            val = str(params["fan_speed"])
            if val in ["auto", "1", "2", "3"]:
                self.fan_speed = val
                changes.append(f"fan speed set to {self.fan_speed}")
                should_turn_on = True
            else:
                return False, "Invalid fan speed."

        if "sleep_mode" in params:
            if params["sleep_mode"] in ["on", "off"]:
                self.sleep_mode = params["sleep_mode"]
                changes.append(f"sleep mode {self.sleep_mode}")
                should_turn_on = True
            else:
                return False, "Invalid sleep mode."

        if "temperature" in params:
            try:
                t = int(params["temperature"])
                if 16 <= t <= 30:
                    self.temperature = t
                    changes.append(f"temperature set to {self.temperature}")
                    should_turn_on = True
                else:
                    return False, "Temperature must be between 16 and 30."
            except ValueError:
                return False, "Temperature must be an integer."

        # --- 2. 定时器逻辑 (Timer) ---
        if "timer" in params:
            try:
                t = float(params["timer"])
                if t < 0:
                    return False, "Timer cannot be negative."
                if t > 5:
                    return False, "Timer cannot exceed 5 hours."
                if t != 0 and t % 0.5 != 0:
                    return False, "Timer must be a multiple of 0.5 hours."

                if t == 0:
                    self.timer = 0
                    changes.append("timer cancelled")
                else:
                    self.timer = t
                    changes.append(f"timer set to {t}h")
                    should_turn_on = True
            except ValueError:
                return False, "Timer must be a number."

        # --- 3. 统一处理自动开机 ---
        if should_turn_on and self.power != "on":
            self.power = "on"
            changes.append("automatically turned on")

        if not changes:
            return True, f"{self.name}: No changes made."
        
        details = ", ".join(changes)
        return True, f"{self.name}: {details}."


# ==================== Test Cases ====================

def test_air_conditioner_basic():
    """Test AirConditioner basic functionality"""
    print("\n=== Testing AirConditioner Basic ===")

    # Test 1: Create AC
    ac = AirConditioner("Test AC")
    print("Test 1: Create AC")
    print(f"  Initial state: {ac.read()}")
    assert ac.power == "off"
    assert ac.mode == "cooling"
    assert ac.fan_speed == "auto"
    assert ac.sleep_mode == "off"
    assert ac.temperature == 26
    assert ac.timer == 0
    print("  ✓ Passed")

    # Test 2: Turn on
    print("\nTest 2: Turn on AC")
    success, msg = ac.update({"power": "on"})
    print(f"  Result: {msg}")
    assert success == True
    assert ac.power == "on"
    print("  ✓ Passed")

    # Test 3: Turn off
    print("\nTest 3: Turn off AC")
    ac.update({"power": "on"})
    ac.timer = 5  # Set a timer
    success, msg = ac.update({"power": "off"})
    print(f"  Result: {msg}")
    assert success == True
    assert ac.power == "off"
    assert ac.timer == 0  # Timer should be cancelled
    print("  ✓ Passed")

    # Test 4: Invalid power state
    print("\nTest 4: Invalid power state")
    success, msg = ac.update({"power": "standby"})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")


def test_air_conditioner_modes():
    """Test AirConditioner mode changes"""
    print("\n=== Testing AirConditioner Modes ===")

    ac = AirConditioner("Test AC")

    # Test 1: All valid modes (should auto turn on)
    print("Test 1: Set all valid modes")
    modes = ["cooling", "heating", "dehumidify"]
    for mode in modes:
        ac.power = "off"
        success, msg = ac.update({"mode": mode})
        print(f"  Set to {mode}: {msg}")
        assert success == True
        assert ac.mode == mode
        assert ac.power == "on"  # Should auto turn on
    print("  ✓ Passed")

    # Test 2: Invalid mode
    print("\nTest 2: Try invalid mode")
    ac.power = "off"
    success, msg = ac.update({"mode": "fan_only"})
    print(f"  Result: {msg}")
    assert success == False
    assert ac.power == "off"  # Should NOT turn on
    print("  ✓ Passed")


def test_air_conditioner_fan_speed():
    """Test AirConditioner fan speed"""
    print("\n=== Testing AirConditioner Fan Speed ===")

    ac = AirConditioner("Test AC")

    # Test 1: All valid fan speeds
    print("Test 1: Set all valid fan speeds")
    fan_speeds = ["auto", "1", "2", "3"]
    for speed in fan_speeds:
        ac.power = "off"
        success, msg = ac.update({"fan_speed": speed})
        print(f"  Set to {speed}: {msg}")
        assert success == True
        assert ac.fan_speed == str(speed)
        assert ac.power == "on"  # Should auto turn on
    print("  ✓ Passed")

    # Test 2: Invalid fan speed
    print("\nTest 2: Try invalid fan speed")
    ac.power = "off"
    success, msg = ac.update({"fan_speed": "5"})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")


def test_air_conditioner_temperature():
    """Test AirConditioner temperature control"""
    print("\n=== Testing AirConditioner Temperature ===")

    ac = AirConditioner("Test AC")

    # Test 1: Valid temperature range (16-30)
    print("Test 1: Test valid temperature range")
    for temp in [16, 20, 26, 30]:
        ac.power = "off"
        success, msg = ac.update({"temperature": temp})
        print(f"  Set to {temp}°C: {msg}")
        assert success == True
        assert ac.temperature == temp
        assert ac.power == "on"  # Should auto turn on
    print("  ✓ Passed")

    # Test 2: Too low temperature
    print("\nTest 2: Try temperature too low (15°C)")
    ac.power = "off"
    success, msg = ac.update({"temperature": 15})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    # Test 3: Too high temperature
    print("\nTest 3: Try temperature too high (31°C)")
    ac.power = "off"
    success, msg = ac.update({"temperature": 31})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    # Test 4: Invalid temperature type
    print("\nTest 4: Try non-integer temperature")
    ac.power = "off"
    success, msg = ac.update({"temperature": "twenty"})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")


def test_air_conditioner_sleep_mode():
    """Test AirConditioner sleep mode"""
    print("\n=== Testing AirConditioner Sleep Mode ===")

    ac = AirConditioner("Test AC")

    # Test 1: Turn on sleep mode (should auto turn on AC)
    print("Test 1: Turn on sleep mode")
    ac.power = "off"
    success, msg = ac.update({"sleep_mode": "on"})
    print(f"  Result: {msg}")
    assert success == True
    assert ac.sleep_mode == "on"
    assert ac.power == "on"
    print("  ✓ Passed")

    # Test 2: Turn off sleep mode
    print("\nTest 2: Turn off sleep mode")
    success, msg = ac.update({"sleep_mode": "off"})
    print(f"  Result: {msg}")
    assert success == True
    assert ac.sleep_mode == "off"
    print("  ✓ Passed")

    # Test 3: Invalid sleep mode
    print("\nTest 3: Try invalid sleep mode")
    success, msg = ac.update({"sleep_mode": "maybe"})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")


def test_air_conditioner_timer():
    """Test AirConditioner timer functionality"""
    print("\n=== Testing AirConditioner Timer ===")

    ac = AirConditioner("Test AC")

    # Test 1: Set timer (should auto turn on)
    print("Test 1: Set timer for 2 hours")
    ac.power = "off"
    ac.timer = 0
    success, msg = ac.update({"timer": 2.0})
    print(f"  Result: {msg}")
    assert success == True
    assert ac.timer == 2.0
    assert ac.power == "on"
    print("  ✓ Passed")

    # Test 2: Timer tick - advance time
    print("\nTest 2: Timer tick - advance 1 hour")
    ac.tick(1.0)
    print(f"  Timer remaining: {ac.timer}h")
    assert ac.timer == 1.0
    assert ac.power == "on"
    print("  ✓ Passed")

    # Test 3: Timer expires
    print("\nTest 3: Timer expires after 1 more hour")
    ac.tick(1.0)
    print(f"  Timer remaining: {ac.timer}h, Power: {ac.power}")
    assert ac.timer == 0
    assert ac.power == "off"  # Should auto turn off
    print("  ✓ Passed")

    # Test 4: Cancel timer
    print("\nTest 4: Cancel timer")
    ac.power = "on"
    ac.timer = 5
    success, msg = ac.update({"timer": 0})
    print(f"  Result: {msg}")
    assert success == True
    assert ac.timer == 0
    print("  ✓ Passed")

    # Test 5: Negative timer
    print("\nTest 5: Try negative timer")
    success, msg = ac.update({"timer": -1})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")


def test_air_conditioner_combined():
    """Test AirConditioner combined settings"""
    print("\n=== Testing AirConditioner Combined Settings ===")

    ac = AirConditioner("Test AC")

    # Test 1: Multiple settings at once
    print("Test 1: Set multiple parameters")
    ac.power = "off"
    success, msg = ac.update({
        "power": "on",
        "mode": "cooling",
        "temperature": 24,
        "fan_speed": "2",
        "sleep_mode": "off"
    })
    print(f"  Result: {msg}")
    assert success == True
    assert ac.power == "on"
    assert ac.mode == "cooling"
    assert ac.temperature == 24
    assert ac.fan_speed == "2"
    assert ac.sleep_mode == "off"
    print("  ✓ Passed")


def test_air_conditioner_broken():
    """Test AirConditioner in broken state"""
    print("\n=== Testing AirConditioner Broken State ===")

    ac = AirConditioner("Test AC")
    ac.break_device()

    print("Test 1: Read broken device")
    state = ac.read()
    print(f"  Result: {state}")
    assert state.get("error") == "ConnectionError"
    print("  ✓ Passed")

    print("\nTest 2: Update broken device")
    success, msg = ac.update({"power": "on"})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    print("\nTest 3: Repair and use again")
    ac.repair()
    success, msg = ac.update({"power": "on"})
    print(f"  Result: {msg}")
    assert success == True
    print("  ✓ Passed")


if __name__ == "__main__":
    test_air_conditioner_basic()
    test_air_conditioner_modes()
    test_air_conditioner_fan_speed()
    test_air_conditioner_temperature()
    test_air_conditioner_sleep_mode()
    test_air_conditioner_timer()
    test_air_conditioner_combined()
    test_air_conditioner_broken()
    print("\n=== All AirConditioner tests passed! ===")