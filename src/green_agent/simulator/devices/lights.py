from ..base import SmartDevice


class Light(SmartDevice):
    def __init__(self, name, has_color=False):
        super().__init__(name)
        self.has_color = has_color
        self.power = "off"
        self.color = "white" if has_color else None

        # 颜色轮转序列（用于自动切换）
        self.color_cycle = ["white", "warm", "blue", "red"]

        # 当前颜色索引（用于轮转）
        self.color_index = 0

    def read(self):
        base = super().read()
        base.update({"power": self.power})
        if self.has_color:
            base["color"] = self.color
        return base

    def update(self, params):
        success, msg = super().update(params)
        if not success: return success, msg

        changes = []

        if "power" in params:
            if params["power"] in ["on", "off"]:
                self.power = params["power"]
                changes.append(f"{self.name} turned {self.power}")

                # 自动颜色轮转：只开灯，没有指定颜色
                if params["power"] == "on" and "color" not in params and self.has_color:
                    next_color = self.color_cycle[self.color_index]
                    self.color = next_color
                    changes.append(f"color auto-cycled to {self.color}")
                    # 更新索引到下一个位置
                    self.color_index = (self.color_index + 1) % len(self.color_cycle)
            else:
                return False, "Invalid power state (on/off)."

        if "color" in params:
            if not self.has_color:
                return False, f"{self.name} does not support color changes."

            valid_colors = ['white', 'red', 'blue', 'warm']
            if params["color"] in valid_colors:
                self.color = params["color"]
                changes.append(f"{self.name} color set to {self.color}")

                # 更新轮转索引到指定颜色的下一个位置
                if self.color in self.color_cycle:
                    color_idx = self.color_cycle.index(self.color)
                    self.color_index = (color_idx + 1) % len(self.color_cycle)

                if self.power != "on":
                    self.power = "on"
                    changes.append("automatically turned on")
            else:
                return False, f"Invalid color. Options: {valid_colors}"

        if not changes:
            return True, f"{self.name}: No changes made."

        details = ", ".join(changes)
        return True, f"{self.name}: {details}."


# ==================== Test Cases ====================

def test_light():
    """Test Light device functionality"""
    print("\n=== Testing Light Device ===")

    # Test 1: Create light without color
    light = Light("Test Light", has_color=False)
    print("Test 1: Create light without color")
    print(f"  Initial state: {light.read()}")
    assert light.power == "off"
    assert light.has_color == False
    assert light.color is None
    print("  ✓ Passed")

    # Test 2: Turn on
    print("\nTest 2: Turn on light")
    success, msg = light.update({"power": "on"})
    print(f"  Result: {msg}")
    assert success == True
    assert light.power == "on"
    state = light.read()
    assert state["power"] == "on"
    print("  ✓ Passed")

    # Test 3: Turn off
    print("\nTest 3: Turn off light")
    success, msg = light.update({"power": "off"})
    print(f"  Result: {msg}")
    assert success == True
    assert light.power == "off"
    print("  ✓ Passed")

    # Test 4: Invalid power state
    print("\nTest 4: Invalid power state")
    success, msg = light.update({"power": "invalid"})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    # Test 5: Color not supported
    print("\nTest 5: Try to change color on non-colored light")
    success, msg = light.update({"color": "red"})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")


def test_colored_light():
    """Test colored Light device functionality"""
    print("\n=== Testing Colored Light Device ===")

    # Test 1: Create colored light
    light = Light("Colored Light", has_color=True)
    print("Test 1: Create colored light")
    print(f"  Initial state: {light.read()}")
    assert light.has_color == True
    assert light.color == "white"
    print("  ✓ Passed")

    # Test 2: Set color while off (should auto turn on)
    print("\nTest 2: Set color to red while off")
    success, msg = light.update({"color": "red"})
    print(f"  Result: {msg}")
    assert success == True
    assert light.color == "red"
    assert light.power == "on"  # Should auto turn on
    print("  ✓ Passed")

    # Test 3: Set all valid colors
    print("\nTest 3: Test all valid colors")
    colors = ['white', 'red', 'blue', 'warm']
    for color in colors:
        success, msg = light.update({"color": color})
        print(f"  Set to {color}: {msg}")
        assert success == True
        assert light.color == color
    print("  ✓ Passed")

    # Test 4: Invalid color
    print("\nTest 4: Try invalid color")
    success, msg = light.update({"color": "purple"})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    # Test 5: Power and color together
    print("\nTest 5: Turn off and set to blue")
    light.update({"power": "off"})
    assert light.power == "off"
    success, msg = light.update({"power": "on", "color": "blue"})
    print(f"  Result: {msg}")
    assert success == True
    assert light.power == "on"
    assert light.color == "blue"
    print("  ✓ Passed")


def test_light_broken():
    """Test Light in broken state"""
    print("\n=== Testing Light Broken State ===")

    light = Light("Test Light", has_color=True)
    light.break_device()

    print("Test 1: Read broken device")
    state = light.read()
    print(f"  Result: {state}")
    assert state.get("error") == "ConnectionError"
    print("  ✓ Passed")

    print("\nTest 2: Update broken device")
    success, msg = light.update({"power": "on"})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    print("\nTest 3: Repair and use again")
    light.repair()
    success, msg = light.update({"power": "on"})
    print(f"  Result: {msg}")
    assert success == True
    print("  ✓ Passed")


if __name__ == "__main__":
    test_light()
    test_colored_light()
    test_light_broken()
    print("\n=== All Light tests passed! ===")