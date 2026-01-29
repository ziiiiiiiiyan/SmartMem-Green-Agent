from ..base import SmartDevice


class SecuritySystem(SmartDevice):
    def __init__(self, name):
        super().__init__(name)
        self.door_lock = "locked" # closed/locked
        self.intercom_msg = ""

    def read(self):
        base = super().read()
        base.update({"door_lock": self.door_lock, "last_intercom_msg": self.intercom_msg})
        return base

    def update(self, params):
        success, msg = super().update(params)
        if not success: return success, msg
        
        if "door_lock" in params:
            if params["door_lock"] in ["open", "closed"]: 
                self.door_lock = params["door_lock"]
                return True, f"door_look set to {self.door_lock}"
            else: 
                return False, "Door lock must be 'open' or 'closed'."
        
        if "intercom_reply" in params:
            self.intercom_msg = str(params["intercom_reply"])
            return True, f"Reply sent to intercom: {self.intercom_msg}"

        return True, f"{self.name}: No changes made."


# ==================== Test Cases ====================

def test_security_system_door_lock():
    """Test SecuritySystem door lock functionality"""
    print("\n=== Testing SecuritySystem Door Lock ===")

    # Test 1: Create security system
    security = SecuritySystem("Test Security")
    print("Test 1: Create security system")
    print(f"  Initial state: {security.read()}")
    assert security.door_lock == "locked"
    assert security.intercom_msg == ""
    print("  ✓ Passed")

    # Test 2: Unlock door
    print("\nTest 2: Unlock door")
    success, msg = security.update({"door_lock": "open"})
    print(f"  Result: {msg}")
    assert success == True
    assert security.door_lock == "open"
    state = security.read()
    assert state["door_lock"] == "open"
    print("  ✓ Passed")

    # Test 3: Close door
    print("\nTest 3: Close door")
    success, msg = security.update({"door_lock": "closed"})
    print(f"  Result: {msg}")
    assert success == True
    assert security.door_lock == "closed"
    print("  ✓ Passed")

    # Test 4: Lock door (set to closed/locked state)
    print("\nTest 4: Lock door")
    success, msg = security.update({"door_lock": "closed"})
    print(f"  Result: {msg}")
    assert success == True
    assert security.door_lock == "closed"
    print("  ✓ Passed")

    # Test 5: Invalid door lock state
    print("\nTest 5: Try invalid door lock state")
    success, msg = security.update({"door_lock": "unlocked"})
    print(f"  Result: {msg}")
    assert success == False
    assert security.door_lock == "closed"  # Should remain unchanged
    print("  ✓ Passed")


def test_security_system_intercom():
    """Test SecuritySystem intercom functionality"""
    print("\n=== Testing SecuritySystem Intercom ===")

    security = SecuritySystem("Test Security")

    # Test 1: Send intercom reply
    print("Test 1: Send intercom reply")
    success, msg = security.update({"intercom_reply": "Hello, who is it?"})
    print(f"  Result: {msg}")
    assert success == True
    assert security.intercom_msg == "Hello, who is it?"
    state = security.read()
    assert state["last_intercom_msg"] == "Hello, who is it?"
    print("  ✓ Passed")

    # Test 2: Another intercom message
    print("\nTest 2: Send another intercom reply")
    success, msg = security.update({"intercom_reply": "Delivery person, leave at door"})
    print(f"  Result: {msg}")
    assert success == True
    assert security.intercom_msg == "Delivery person, leave at door"
    print("  ✓ Passed")

    # Test 3: Numeric intercom reply (should convert to string)
    print("\nTest 3: Send numeric intercom reply")
    success, msg = security.update({"intercom_reply": 12345})
    print(f"  Result: {msg}")
    assert success == True
    assert security.intercom_msg == "12345"
    print("  ✓ Passed")


def test_security_system_combined():
    """Test SecuritySystem combined operations"""
    print("\n=== Testing SecuritySystem Combined Operations ===")

    security = SecuritySystem("Test Security")

    # Test 1: Update both door lock and intercom
    print("Test 1: Try multiple updates (only first will be processed)")
    # Note: The current implementation only processes one parameter at a time
    success, msg = security.update({
        "door_lock": "open",
        "intercom_reply": "Come in"
    })
    print(f"  Result: {msg}")
    assert success == True
    # Only door_lock will be updated
    assert security.door_lock == "open"
    print("  ✓ Passed")

    # Test 2: Update intercom separately
    print("\nTest 2: Update intercom separately")
    success, msg = security.update({"intercom_reply": "Welcome home"})
    print(f"  Result: {msg}")
    assert success == True
    assert security.intercom_msg == "Welcome home"
    print("  ✓ Passed")


def test_security_system_broken():
    """Test SecuritySystem in broken state"""
    print("\n=== Testing SecuritySystem Broken State ===")

    security = SecuritySystem("Test Security")
    security.break_device()

    print("Test 1: Read broken device")
    state = security.read()
    print(f"  Result: {state}")
    assert state.get("error") == "ConnectionError"
    print("  ✓ Passed")

    print("\nTest 2: Update broken device - door lock")
    success, msg = security.update({"door_lock": "open"})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    print("\nTest 3: Update broken device - intercom")
    success, msg = security.update({"intercom_reply": "Test message"})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    print("\nTest 4: Repair and use again")
    security.repair()
    success, msg = security.update({"door_lock": "open"})
    print(f"  Result: {msg}")
    assert success == True
    print("  ✓ Passed")


if __name__ == "__main__":
    test_security_system_door_lock()
    test_security_system_intercom()
    test_security_system_combined()
    test_security_system_broken()
    print("\n=== All SecuritySystem tests passed! ===")