from ..base import SmartDevice


class WashingMachine(SmartDevice):
    def __init__(self, name):
        super().__init__(name)
        self.schedule_timer = 0.0 
        self.run_timer = 0.0
        self.current_mode = None
        self.status = "idle"

    def is_busy(self):
        return self.status in ["scheduled", "running"]

    def tick(self, hours):
        time_passed = hours
        
        # 1. 处理预约倒计时
        if self.status == "scheduled":
            if self.schedule_timer > time_passed:
                self.schedule_timer -= time_passed
                time_passed = 0
            else:
                time_passed -= self.schedule_timer
                self.schedule_timer = 0
                self.status = "running"
        
        # 2. 处理运行倒计时
        if self.status == "running":
            if self.run_timer > 0:
                self.run_timer = max(0, self.run_timer - time_passed)
            
            if self.run_timer == 0:
                self.status = "idle"
                self.current_mode = None

    def read(self):
        base = super().read()
        base.update({
            "status": self.status,
            "mode": self.current_mode,
            "time_until_start": self.schedule_timer,
            "time_until_finish": self.run_timer
        })
        return base

    def update(self, params):
        # 1. 基础检查
        success, msg = super().update(params)
        if not success: return success, msg

        # 限制：只有正在运行中不能改，但预约等待中是可以改的
        if self.status == "running":
            return False, "Machine is running, cannot change settings now."

        changes = []           # 记录本轮操作的所有变动
        mode_updated = False   # 标记是否更新了模式

        # --- 2. 处理模式 (Mode Logic) ---
        if "mode" in params:
            modes = {
                "quick": 0.5, "mix": 1.0, "wool": 1.5, 
                "dry_1h": 1.0, "dry_2h": 2.0, "dry_3h": 3.0
            }
            mode_input = params["mode"]
            duration = 0
            new_mode_name = None

            if mode_input == "dry":
                dry_h = int(params.get("dry_hours", 0))
                if dry_h not in [1, 2, 3]: return False, "Dry hours must be 1, 2, or 3."
                duration = float(dry_h)
                new_mode_name = f"dry ({dry_h}h)"
            elif mode_input in modes:
                duration = modes[mode_input]
                new_mode_name = mode_input
            else:
                return False, f"Invalid mode. Options: {list(modes.keys())} or 'dry'."

            # 应用模式更新
            self.current_mode = new_mode_name
            self.run_timer = duration
            mode_updated = True
            changes.append(f"mode set to {new_mode_name} ({duration}h)")

        # --- 3. 处理时间/预约 (Schedule Logic) ---
        if "schedule_hours" in params:
            try:
                h = int(params["schedule_hours"])
                if not (1 <= h <= 10): return False, "Schedule must be 1-10 hours."
                
                # 检查依赖：如果是单纯改时间，必须保证机器里已经有模式了
                if not self.current_mode:
                    return False, "No previous mode set. Please specify 'mode' to start a schedule."

                self.schedule_timer = h
                self.status = "scheduled"
                
                if not mode_updated:
                    changes.append(f"{self.current_mode} rescheduled to start in {h}h")
                else:
                    changes.append(f"{self.current_mode} scheduled to start in {h}h")
                    
            except ValueError: return False, "Invalid hours format."

        # --- 4. 处理立即开始的情况 ---
        # 如果更新了模式，但没有设置预约时间（且原状态不是预约），则默认为立即开始
        if mode_updated and "schedule_hours" not in params:
            self.status = "running"
            self.schedule_timer = 0
            changes.append("started immediately")

        # --- 5. 生成最终汇总信息 ---
        if not changes:
            return False, "No valid parameters provided (need 'mode' or 'schedule_hours')."

        details = ", ".join(changes)
        return True, f"{self.name}: {details}."
    

class Speaker(SmartDevice):
    def __init__(self, name):
        super().__init__(name)
        self.volume = 5

    def read(self):
        base = super().read()
        base["volume"] = self.volume
        return base

    def update(self, params):
        success, msg = super().update(params)
        if not success: return success, msg

        if "volume" in params:
            try:
                v = int(params["volume"])
                if 0 <= v <= 10: self.volume = v
                else: return False, "Volume must be 0-10."
            except: return False, "Volume must be integer."
        return True, f"Speaker volume set to {self.volume}."


# ==================== Test Cases ====================

def test_washing_machine_basic():
    """Test WashingMachine basic functionality"""
    print("\n=== Testing WashingMachine Basic ===")

    # Test 1: Create washing machine
    wm = WashingMachine("Test Washing Machine")
    print("Test 1: Create washing machine")
    print(f"  Initial state: {wm.read()}")
    assert wm.status == "idle"
    assert wm.current_mode is None
    assert wm.schedule_timer == 0
    assert wm.run_timer == 0
    assert wm.is_busy() == False
    print("  ✓ Passed")


def test_washing_machine_immediate_start():
    """Test WashingMachine immediate start"""
    print("\n=== Testing WashingMachine Immediate Start ===")

    wm = WashingMachine("Test Washing Machine")

    # Test 1: Start quick mode immediately
    print("Test 1: Start quick mode (0.5h)")
    success, msg = wm.update({"mode": "quick"})
    print(f"  Result: {msg}")
    assert success == True
    assert wm.status == "running"
    assert wm.current_mode == "quick"
    assert wm.run_timer == 0.5
    assert wm.is_busy() == True
    print("  ✓ Passed")

    # Test 2: Tick to complete quick wash
    print("\nTest 2: Tick 0.5h to complete quick wash")
    wm.tick(0.5)
    print(f"  Status: {wm.status}, is_busy: {wm.is_busy()}")
    assert wm.status == "idle"
    assert wm.current_mode is None
    assert wm.is_busy() == False
    print("  ✓ Passed")


def test_washing_machine_all_modes():
    """Test WashingMachine all washing modes"""
    print("\n=== Testing WashingMachine All Modes ===")

    wm = WashingMachine("Test Washing Machine")
    modes = {
        "quick": 0.5,
        "mix": 1.0,
        "wool": 1.5,
        "dry_1h": 1.0,
        "dry_2h": 2.0,
        "dry_3h": 3.0
    }

    print("Test 1: Test all valid modes")
    for mode, expected_time in modes.items():
        wm = WashingMachine("Test Washing Machine")
        success, msg = wm.update({"mode": mode})
        print(f"  Mode {mode}: {msg}")
        assert success == True
        assert wm.status == "running"
        assert wm.run_timer == expected_time
    print("  ✓ Passed")

    print("\nTest 2: Test dry mode with hours")
    for h in [1, 2, 3]:
        wm = WashingMachine("Test Washing Machine")
        success, msg = wm.update({"mode": "dry", "dry_hours": h})
        print(f"  dry ({h}h): {msg}")
        assert success == True
        assert wm.status == "running"
        assert wm.run_timer == float(h)
    print("  ✓ Passed")

    print("\nTest 3: Try invalid dry hours")
    wm = WashingMachine("Test Washing Machine")
    success, msg = wm.update({"mode": "dry", "dry_hours": 5})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    print("\nTest 4: Try invalid mode")
    wm = WashingMachine("Test Washing Machine")
    success, msg = wm.update({"mode": "heavy_duty"})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")


def test_washing_machine_schedule():
    """Test WashingMachine scheduling"""
    print("\n=== Testing WashingMachine Scheduling ===")

    wm = WashingMachine("Test Washing Machine")

    # Test 1: Set mode first, then schedule
    print("Test 1: Set mode then schedule for 3 hours")
    success, msg = wm.update({"mode": "quick", "schedule_hours": 3})
    print(f"  Result: {msg}")
    assert success == True
    assert wm.status == "scheduled"
    assert wm.schedule_timer == 3
    assert wm.run_timer == 0.5
    assert wm.is_busy() == True
    print("  ✓ Passed")

    # Test 2: Tick through schedule time
    print("\nTest 2: Tick 2 hours (still scheduled)")
    wm.tick(2.0)
    print(f"  Status: {wm.status}, schedule_timer: {wm.schedule_timer}")
    assert wm.status == "scheduled"
    assert wm.schedule_timer == 1
    print("  ✓ Passed")

    # Test 3: Complete schedule, start running
    print("\nTest 3: Tick 1 more hour (should start running)")
    wm.tick(1.0)
    print(f"  Status: {wm.status}, run_timer: {wm.run_timer}")
    assert wm.status == "running"
    assert wm.schedule_timer == 0
    assert wm.run_timer == 0.5
    print("  ✓ Passed")

    # Test 4: Complete washing
    print("\nTest 4: Complete washing cycle")
    wm.tick(0.5)
    print(f"  Status: {wm.status}, is_busy: {wm.is_busy()}")
    assert wm.status == "idle"
    assert wm.is_busy() == False
    print("  ✓ Passed")

    # Test 5: One-shot mode + schedule
    print("\nTest 5: Set mode and schedule together")
    wm = WashingMachine("Test Washing Machine")
    success, msg = wm.update({"mode": "wool", "schedule_hours": 5})
    print(f"  Result: {msg}")
    assert success == True
    assert wm.status == "scheduled"
    assert wm.schedule_timer == 5
    assert wm.run_timer == 1.5
    print("  ✓ Passed")


def test_washing_machine_running_restrictions():
    """Test WashingMachine restrictions when running"""
    print("\n=== Testing WashingMachine Running Restrictions ===")

    wm = WashingMachine("Test Washing Machine")

    # Test 1: Start a wash
    print("Test 1: Start washing")
    wm.update({"mode": "mix"})
    assert wm.status == "running"
    print("  ✓ Passed")

    # Test 2: Cannot change mode while running
    print("\nTest 2: Try to change mode while running")
    success, msg = wm.update({"mode": "quick"})
    print(f"  Result: {msg}")
    assert success == False
    assert wm.status == "running"
    assert wm.current_mode == "mix"
    print("  ✓ Passed")

    # Test 3: Cannot reschedule while running
    print("\nTest 3: Try to reschedule while running")
    success, msg = wm.update({"schedule_hours": 2})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    # Test 4: Can change mode while scheduled (not running)
    print("\nTest 4: Can change mode while scheduled")
    wm2 = WashingMachine("Test Washing Machine 2")
    wm2.update({"mode": "quick", "schedule_hours": 5})
    assert wm2.status == "scheduled"
    success, msg = wm2.update({"mode": "wool", "schedule_hours": 2})
    print(f"  Result: {msg}")
    assert success == True
    assert wm2.current_mode == "wool"
    assert wm2.schedule_timer == 2
    print("  ✓ Passed")


def test_washing_machine_validation():
    """Test WashingMachine input validation"""
    print("\n=== Testing WashingMachine Validation ===")

    wm = WashingMachine("Test Washing Machine")

    # Test 1: Invalid schedule range (too low)
    print("Test 1: Try schedule_hours=0 (too low)")
    success, msg = wm.update({"mode": "quick", "schedule_hours": 0})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    # Test 2: Invalid schedule range (too high)
    print("\nTest 2: Try schedule_hours=11 (too high)")
    success, msg = wm.update({"mode": "quick", "schedule_hours": 11})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    # Test 3: Schedule without mode
    print("\nTest 3: Try schedule without mode")
    success, msg = wm.update({"schedule_hours": 3})
    print(f"  Result: {msg}")
    assert success == True
    print("  ✓ Passed")


def test_washing_machine_broken():
    """Test WashingMachine in broken state"""
    print("\n=== Testing WashingMachine Broken State ===")

    wm = WashingMachine("Test Washing Machine")
    wm.break_device()

    print("Test 1: Read broken device")
    state = wm.read()
    print(f"  Result: {state}")
    assert state.get("error") == "ConnectionError"
    print("  ✓ Passed")

    print("\nTest 2: Update broken device")
    success, msg = wm.update({"mode": "quick"})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    print("\nTest 3: Repair and use again")
    wm.repair()
    success, msg = wm.update({"mode": "quick"})
    print(f"  Result: {msg}")
    assert success == True
    print("  ✓ Passed")


def test_speaker_basic():
    """Test Speaker basic functionality"""
    print("\n=== Testing Speaker Basic ===")

    # Test 1: Create speaker
    speaker = Speaker("Test Speaker")
    print("Test 1: Create speaker")
    print(f"  Initial state: {speaker.read()}")
    assert speaker.volume == 5
    print("  ✓ Passed")

    # Test 2: Set volume
    print("\nTest 2: Set volume to 8")
    success, msg = speaker.update({"volume": 8})
    print(f"  Result: {msg}")
    assert success == True
    assert speaker.volume == 8
    state = speaker.read()
    assert state["volume"] == 8
    print("  ✓ Passed")


def test_speaker_volume_range():
    """Test Speaker volume range"""
    print("\n=== Testing Speaker Volume Range ===")

    speaker = Speaker("Test Speaker")

    # Test 1: All valid volumes (0-10)
    print("Test 1: Test all valid volumes")
    for vol in range(11):
        success, msg = speaker.update({"volume": vol})
        print(f"  Set to {vol}: {msg}")
        assert success == True
        assert speaker.volume == vol
    print("  ✓ Passed")

    # Test 2: Volume too low
    print("\nTest 2: Try volume -1")
    success, msg = speaker.update({"volume": -1})
    print(f"  Result: {msg}")
    assert success == False
    assert speaker.volume == 10  # Should remain at last valid value
    print("  ✓ Passed")

    # Test 3: Volume too high
    print("\nTest 3: Try volume 11")
    success, msg = speaker.update({"volume": 11})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    # Test 4: Invalid volume type
    print("\nTest 4: Try non-integer volume")
    success, msg = speaker.update({"volume": "high"})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")


def test_speaker_broken():
    """Test Speaker in broken state"""
    print("\n=== Testing Speaker Broken State ===")

    speaker = Speaker("Test Speaker")
    speaker.break_device()

    print("Test 1: Read broken device")
    state = speaker.read()
    print(f"  Result: {state}")
    assert state.get("error") == "ConnectionError"
    print("  ✓ Passed")

    print("\nTest 2: Update broken device")
    success, msg = speaker.update({"volume": 3})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    print("\nTest 3: Repair and use again")
    speaker.repair()
    success, msg = speaker.update({"volume": 7})
    print(f"  Result: {msg}")
    assert success == True
    print("  ✓ Passed")


if __name__ == "__main__":
    # WashingMachine tests
    test_washing_machine_basic()
    test_washing_machine_immediate_start()
    test_washing_machine_all_modes()
    test_washing_machine_schedule()
    test_washing_machine_running_restrictions()
    test_washing_machine_validation()
    test_washing_machine_broken()

    # Speaker tests
    test_speaker_basic()
    test_speaker_volume_range()
    test_speaker_broken()

    print("\n=== All Appliances tests passed! ===")