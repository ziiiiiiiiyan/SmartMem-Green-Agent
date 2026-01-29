from typing import Dict

from ..base import SmartDevice


class Fridge(SmartDevice):
    def __init__(self, name):
        super().__init__(name)
        # Inventory max 10
        self.inventory = {
            "coffee_bean": 10,
            "milk": 10,
            "oat_milk": 10,
            "chocolate_sauce": 10
        }

    def read(self):
        base = super().read()
        base["inventory"] = self.inventory
        return base

    def has_ingredients(self, required: Dict):
        """required: dict of {item: count}"""
        for item, count in required.items():
            if self.inventory.get(item, 0) < count:
                return False, f"Not enough {item}"
        return True, "OK"

    def consume(self, required: Dict):
        for item, count in required.items():
            self.inventory[item] -= count
    
    def set_inventory(self, item: str, count: int):
        if item in self.inventory:
            self.inventory[item] = max(0, min(10, count))


class CoffeeMaker(SmartDevice):
    def __init__(self, name, fridge_instance):
        super().__init__(name)
        self.fridge = fridge_instance # Dependency injection

    def read(self):
        return super().read()

    def update(self, params):
        success, msg = super().update(params)
        if not success: return success, msg

        if "make_coffee" in params:
            recipe_name = params["make_coffee"]
            recipes = {
                "americano": {"coffee_bean": 1},
                "latte": {"coffee_bean": 1, "milk": 2},
                "oat_latte": {"coffee_bean": 1, "oat_milk": 2},
                "mocha": {"coffee_bean": 1, "milk": 1, "chocolate_sauce": 1}
            }

            if recipe_name not in recipes:
                return False, "Unknown recipe."

            required = recipes[recipe_name]
            
            # Check Fridge
            check, check_msg = self.fridge.has_ingredients(required)
            if not check:
                return False, f"Cannot make {recipe_name}: {check_msg}"
            
            # Consume and Make
            self.fridge.consume(required)
            return True, f"Made {recipe_name}. Enjoy!"

        return False, "Please specify 'make_coffee': [recipe_name]."
    
    
class PressureCooker(SmartDevice):
    def __init__(self, name):
        super().__init__(name)
        self.status = "idle"         # idle, scheduled, cooking
        self.remaining_time = 0.0    # 当前阶段的剩余时间（可能是预约倒计时，也可能是烹饪倒计时）
        self.pending_mode = None     # 新增：记录预约的具体模式
        self.pending_duration = 0.0  # 新增：记录预约模式对应的烹饪时长

    def is_busy(self):
        return self.status in ["scheduled", "cooking"]

    def read(self):
        base = super().read()
        base.update({
            "status": self.status,
            "remaining_hours": self.remaining_time,
            "current_mode": self.pending_mode if self.status != "idle" else None
        })
        return base

    def tick(self, hours):
        """
        推进时间。
        逻辑：
        1. 如果是预约状态(scheduled), 先扣减预约时间。如果时间到了, 自动转入烹饪状态(cooking)。
        2. 如果转入烹饪状态后还有剩余时间或者本来就是cooking, 继续扣减烹饪时间。
        3. 烹饪时间归零后，转回 idle。
        """
        if self.status == "idle":
            return

        time_left = hours

        # --- 阶段 1: 处理预约倒计时 ---
        if self.status == "scheduled":
            if self.remaining_time > time_left:
                # 传入的时间不够走完预约
                self.remaining_time -= time_left
                return # 时间耗尽，状态维持 scheduled
            else:
                # 预约时间到了，转进烹饪
                time_left -= self.remaining_time # 扣掉预约消耗的时间，剩下的给烹饪用
                self.status = "cooking"
                self.remaining_time = self.pending_duration # 载入烹饪时长
                # 注意：不要 return，继续往下走，用剩余的 time_left 扣除烹饪时间

        # --- 阶段 2: 处理烹饪倒计时 ---
        if self.status == "cooking":
            if self.remaining_time > time_left:
                self.remaining_time -= time_left
            else:
                # 煮完了
                self.remaining_time = 0.0
                self.status = "idle"
                self.pending_mode = None
                self.pending_duration = 0.0

    def update(self, params):
        # 基础检查
        success, msg = super().update(params)
        if not success: return success, msg

        if self.status == "cooking":
            return False, "Device is currently cooking and cannot be modified."

        modes = {
            "beef_mutton": 1.5,
            "chicken_duck": 1.0,
            "vegetables": 0.5
        }
        
        # 1. 确定烹饪模式 (Mode Resolution)
        # 逻辑：优先用参数里的新模式；如果没有，尝试沿用旧模式；如果旧模式也没有，报错。
        input_mode = params.get("mode")
        final_mode = None
        
        if input_mode:
            if input_mode in modes:
                final_mode = input_mode
            else:
                return False, f"Invalid mode. Options: {', '.join(modes.keys())}."
        else:
            # 用户没传 mode，检查是否有旧模式可用
            if self.pending_mode:
                final_mode = self.pending_mode
            else:
                # 既没传新模式，以前也没设置过，不知道煮什么
                return False, "Cooking mode is required (e.g., beef_mutton)."

        # 2. 预约时间检查 (Schedule Check)
        schedule_delay = 0.0
        if "schedule_hours" in params:
            try:
                h = float(params["schedule_hours"])
                if h < 0:
                    return False, "Time cannot be negative."
                if h > 8.0:
                    return False, "Max schedule time is 8 hours."
                if h % 0.5 != 0:
                    return False, "Schedule time must be a multiple of 0.5 hours."
                schedule_delay = h
            except ValueError:
                return False, "Invalid time format."
        
        # 3. 更新状态
        cooking_duration = modes[final_mode]
        self.pending_mode = final_mode
        self.pending_duration = cooking_duration
        
        # 文案优化：如果是单纯改时间（没传mode且本来就在预约中），用 Rescheduled 提示
        is_reschedule = (not input_mode) and (schedule_delay > 0)
        action_verb = "Rescheduled" if is_reschedule else "Scheduled"
        
        if schedule_delay > 0:
            # ---> 进入/重置为 预约模式
            self.status = "scheduled"
            self.remaining_time = schedule_delay
            return True, f"{action_verb} {final_mode} to start in {schedule_delay}h."
        else:
            # ---> 立即开始
            # (如果之前是预约状态，这里 schedule_delay=0 意味着取消预约立即开始)
            self.status = "cooking"
            self.remaining_time = cooking_duration
            return True, f"Started cooking {final_mode} ({cooking_duration}h)."


# ==================== Test Cases ====================

def test_fridge_basic():
    """Test Fridge basic functionality"""
    print("\n=== Testing Fridge Basic ===")

    # Test 1: Create fridge
    fridge = Fridge("Test Fridge")
    print("Test 1: Create fridge")
    print(f"  Initial state: {fridge.read()}")
    assert fridge.inventory == {
        "coffee_bean": 10,
        "milk": 10,
        "oat_milk": 10,
        "chocolate_sauce": 10
    }
    print("  ✓ Passed")

    # Test 2: Read inventory
    print("\nTest 2: Read inventory")
    state = fridge.read()
    print(f"  Inventory: {state['inventory']}")
    assert state["inventory"]["coffee_bean"] == 10
    assert state["inventory"]["milk"] == 10
    print("  ✓ Passed")


def test_fridge_inventory_management():
    """Test Fridge inventory management"""
    print("\n=== Testing Fridge Inventory Management ===")

    fridge = Fridge("Test Fridge")

    # Test 1: Check has_ingredients - sufficient
    print("Test 1: Check has_ingredients - sufficient")
    success, msg = fridge.has_ingredients({"coffee_bean": 2, "milk": 1})
    print(f"  Result: {msg}")
    assert success == True
    print("  ✓ Passed")

    # Test 2: Check has_ingredients - insufficient
    print("\nTest 2: Check has_ingredients - insufficient")
    success, msg = fridge.has_ingredients({"coffee_bean": 15})
    print(f"  Result: {msg}")
    assert success == False
    assert "coffee_bean" in msg
    print("  ✓ Passed")

    # Test 3: Consume ingredients
    print("\nTest 3: Consume ingredients")
    initial_milk = fridge.inventory["milk"]
    fridge.consume({"milk": 2})
    assert fridge.inventory["milk"] == initial_milk - 2
    print(f"  Milk: {initial_milk} -> {fridge.inventory['milk']}")
    print("  ✓ Passed")

    # Test 4: Set inventory
    print("\nTest 4: Set inventory")
    fridge.set_inventory("coffee_bean", 5)
    assert fridge.inventory["coffee_bean"] == 5
    print(f"  Coffee bean set to: {fridge.inventory['coffee_bean']}")
    print("  ✓ Passed")

    # Test 5: Set inventory to max (10)
    print("\nTest 5: Set inventory to max (10)")
    fridge.set_inventory("milk", 15)
    assert fridge.inventory["milk"] == 10  # Capped at 10
    print(f"  Milk capped at: {fridge.inventory['milk']}")
    print("  ✓ Passed")

    # Test 6: Set inventory to min (0)
    print("\nTest 6: Set inventory to min (0)")
    fridge.set_inventory("oat_milk", -5)
    assert fridge.inventory["oat_milk"] == 0  # Floored at 0
    print(f"  Oat milk floored at: {fridge.inventory['oat_milk']}")
    print("  ✓ Passed")


def test_fridge_broken():
    """Test Fridge in broken state"""
    print("\n=== Testing Fridge Broken State ===")

    fridge = Fridge("Test Fridge")
    fridge.break_device()

    print("Test 1: Read broken fridge")
    state = fridge.read()
    print(f"  Result: {state}")
    assert state.get("error") == "ConnectionError"
    print("  ✓ Passed")

    print("\nTest 2: Broken fridge still has inventory (data intact)")
    # Note: The broken state affects connection, but data is preserved
    assert fridge.inventory["coffee_bean"] == 10
    print("  ✓ Passed")

    print("\nTest 3: Repair and read again")
    fridge.repair()
    state = fridge.read()
    assert state.get("error") is None
    assert state["inventory"]["coffee_bean"] == 10
    print("  ✓ Passed")


def test_coffee_maker_basic():
    """Test CoffeeMaker basic functionality"""
    print("\n=== Testing CoffeeMaker Basic ===")

    # Test 1: Create coffee maker with fridge
    fridge = Fridge("Test Fridge")
    coffee_maker = CoffeeMaker("Test Coffee Maker", fridge)
    print("Test 1: Create coffee maker with fridge")
    print(f"  Initial state: {coffee_maker.read()}")
    assert coffee_maker.fridge == fridge
    print("  ✓ Passed")


def test_coffee_maker_recipes():
    """Test CoffeeMaker all recipes"""
    print("\n=== Testing CoffeeMaker Recipes ===")

    fridge = Fridge("Test Fridge")
    coffee_maker = CoffeeMaker("Test Coffee Maker", fridge)

    # Test 1: Make americano
    print("Test 1: Make americano (1 coffee_bean)")
    success, msg = coffee_maker.update({"make_coffee": "americano"})
    print(f"  Result: {msg}")
    assert success == True
    assert fridge.inventory["coffee_bean"] == 9
    print("  ✓ Passed")

    # Test 2: Make latte
    print("\nTest 2: Make latte (1 coffee_bean, 2 milk)")
    success, msg = coffee_maker.update({"make_coffee": "latte"})
    print(f"  Result: {msg}")
    assert success == True
    assert fridge.inventory["coffee_bean"] == 8
    assert fridge.inventory["milk"] == 8
    print("  ✓ Passed")

    # Test 3: Make oat latte
    print("\nTest 3: Make oat latte (1 coffee_bean, 2 oat_milk)")
    success, msg = coffee_maker.update({"make_coffee": "oat_latte"})
    print(f"  Result: {msg}")
    assert success == True
    assert fridge.inventory["coffee_bean"] == 7
    assert fridge.inventory["oat_milk"] == 8
    print("  ✓ Passed")

    # Test 4: Make mocha
    print("\nTest 4: Make mocha (1 coffee_bean, 1 milk, 1 chocolate_sauce)")
    success, msg = coffee_maker.update({"make_coffee": "mocha"})
    print(f"  Result: {msg}")
    assert success == True
    assert fridge.inventory["coffee_bean"] == 6
    assert fridge.inventory["milk"] == 7
    assert fridge.inventory["chocolate_sauce"] == 9
    print("  ✓ Passed")


def test_coffee_maker_insufficient_ingredients():
    """Test CoffeeMaker with insufficient ingredients"""
    print("\n=== Testing CoffeeMaker Insufficient Ingredients ===")

    fridge = Fridge("Test Fridge")
    coffee_maker = CoffeeMaker("Test Coffee Maker", fridge)

    # Test 1: Deplete milk
    print("Test 1: Deplete milk")
    fridge.set_inventory("milk", 0)
    success, msg = coffee_maker.update({"make_coffee": "latte"})
    print(f"  Result: {msg}")
    assert success == False
    assert "milk" in msg
    print("  ✓ Passed")

    # Test 2: Try unknown recipe
    print("\nTest 2: Try unknown recipe")
    success, msg = coffee_maker.update({"make_coffee": "cappuccino"})
    print(f"  Result: {msg}")
    assert success == False
    assert "Unknown" in msg
    print("  ✓ Passed")


def test_coffee_maker_broken():
    """Test CoffeeMaker in broken state"""
    print("\n=== Testing CoffeeMaker Broken State ===")

    fridge = Fridge("Test Fridge")
    coffee_maker = CoffeeMaker("Test Coffee Maker", fridge)
    coffee_maker.break_device()

    print("Test 1: Read broken device")
    state = coffee_maker.read()
    print(f"  Result: {state}")
    assert state.get("error") == "ConnectionError"
    print("  ✓ Passed")

    print("\nTest 2: Cannot make coffee when broken")
    success, msg = coffee_maker.update({"make_coffee": "americano"})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    print("\nTest 3: Repair and make coffee")
    coffee_maker.repair()
    success, msg = coffee_maker.update({"make_coffee": "americano"})
    print(f"  Result: {msg}")
    assert success == True
    print("  ✓ Passed")


def test_pressure_cooker_basic():
    """Test PressureCooker basic functionality"""
    print("\n=== Testing PressureCooker Basic ===")

    # Test 1: Create pressure cooker
    cooker = PressureCooker("Test Pressure Cooker")
    print("Test 1: Create pressure cooker")
    print(f"  Initial state: {cooker.read()}")
    assert cooker.status == "idle"
    assert cooker.remaining_time == 0
    assert cooker.pending_mode is None
    assert cooker.pending_duration == 0
    assert cooker.is_busy() == False
    print("  ✓ Passed")


def test_pressure_cooker_modes():
    """Test PressureCooker cooking modes"""
    print("\n=== Testing PressureCooker Cooking Modes ===")

    modes = {
        "beef_mutton": 1.5,
        "chicken_duck": 1.0,
        "vegetables": 0.5
    }

    print("Test 1: Test all valid modes")
    for mode, expected_time in modes.items():
        cooker = PressureCooker("Test Pressure Cooker")
        success, msg = cooker.update({"mode": mode})
        print(f"  Mode {mode}: {msg}")
        assert success == True
        assert cooker.status == "cooking"
        assert cooker.remaining_time == expected_time
        assert cooker.pending_mode == mode
        assert cooker.pending_duration == expected_time
        assert cooker.is_busy() == True
        state = cooker.read()
        assert state["current_mode"] == mode
    print("  ✓ Passed")

    print("\nTest 2: Try invalid mode")
    cooker = PressureCooker("Test Pressure Cooker")
    success, msg = cooker.update({"mode": "soup"})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")


def test_pressure_cooker_schedule():
    """Test PressureCooker scheduling"""
    print("\n=== Testing PressureCooker Scheduling ===")

    cooker = PressureCooker("Test Pressure Cooker")

    # Test 1: Schedule cooking
    print("Test 1: Schedule vegetables for 2 hours")
    success, msg = cooker.update({"mode": "vegetables", "schedule_hours": 2.0})
    print(f"  Result: {msg}")
    assert success == True
    assert cooker.status == "scheduled"
    assert cooker.remaining_time == 2.0
    assert cooker.pending_mode == "vegetables"
    assert cooker.pending_duration == 0.5
    assert cooker.is_busy() == True
    print("  ✓ Passed")

    # Test 2: Tick through schedule
    print("\nTest 2: Tick 1.5 hours (still scheduled)")
    cooker.tick(1.5)
    print(f"  Remaining: {cooker.remaining_time}h")
    assert cooker.remaining_time == 0.5
    assert cooker.status == "scheduled"
    print("  ✓ Passed")

    # Test 3: Complete schedule, should transition to cooking
    print("\nTest 3: Complete schedule (tick 0.5h), should start cooking")
    cooker.tick(0.5)
    print(f"  Status: {cooker.status}, remaining_time: {cooker.remaining_time}h")
    assert cooker.status == "cooking"  # Changed: now goes to cooking, not idle
    assert cooker.remaining_time == 0.5  # pending_duration for vegetables
    assert cooker.is_busy() == True
    print("  ✓ Passed")

    # Test 4: Complete cooking
    print("\nTest 4: Complete cooking (tick 0.5h)")
    cooker.tick(0.5)
    print(f"  Status: {cooker.status}")
    assert cooker.status == "idle"
    assert cooker.remaining_time == 0
    assert cooker.pending_mode is None
    assert cooker.is_busy() == False
    print("  ✓ Passed")


def test_pressure_cooker_schedule_validation():
    """Test PressureCooker schedule validation"""
    print("\n=== Testing PressureCooker Schedule Validation ===")

    # Test 1: Valid schedule times (multiples of 0.5)
    print("Test 1: Test valid schedule times")
    for h in [0.5, 1.0, 2.5, 8.0]:
        cooker = PressureCooker("Test Pressure Cooker")
        success, msg = cooker.update({"mode": "vegetables", "schedule_hours": h})
        assert success == True
        assert cooker.status == "scheduled"
        print(f"  {h}h: OK")
    print("  ✓ Passed")

    # Test 2: schedule_hours=0 with mode (should start cooking immediately)
    print("\nTest 2: schedule_hours=0 with mode (start immediately)")
    cooker = PressureCooker("Test Pressure Cooker")
    success, msg = cooker.update({"mode": "chicken_duck", "schedule_hours": 0})
    print(f"  Result: {msg}")
    assert success == True
    assert cooker.status == "cooking"  # Should start cooking immediately
    assert cooker.remaining_time == 1.0
    print("  ✓ Passed")

    # Test 3: schedule_hours=0 without mode (should fail)
    print("\nTest 3: schedule_hours=0 without mode (should fail)")
    cooker = PressureCooker("Test Pressure Cooker")
    success, msg = cooker.update({"schedule_hours": 0})
    print(f"  Result: {msg}")
    assert success == False
    assert "mode is required" in msg.lower()
    print("  ✓ Passed")

    # Test 4: Invalid - not multiple of 0.5
    print("\nTest 4: Try invalid schedule (1.3h)")
    cooker = PressureCooker("Test Pressure Cooker")
    success, msg = cooker.update({"mode": "vegetables", "schedule_hours": 1.3})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    # Test 5: Invalid - too high
    print("\nTest 5: Try too high schedule (9h)")
    success, msg = cooker.update({"mode": "vegetables", "schedule_hours": 9.0})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    # Test 6: Invalid - negative
    print("\nTest 6: Try negative schedule")
    success, msg = cooker.update({"mode": "vegetables", "schedule_hours": -1.0})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")


def test_pressure_cooker_busy():
    """Test PressureCooker when busy"""
    print("\n=== Testing PressureCooker Busy State ===")

    cooker = PressureCooker("Test Pressure Cooker")

    # Test 1: Start cooking
    print("Test 1: Start cooking")
    cooker.update({"mode": "chicken_duck"})
    assert cooker.status == "cooking"
    assert cooker.is_busy() == True
    print("  ✓ Passed")

    # Test 2: Cannot change mode when cooking
    print("\nTest 2: Cannot change mode when cooking")
    success, msg = cooker.update({"mode": "vegetables"})
    print(f"  Result: {msg}")
    assert success == False
    assert cooker.status == "cooking"
    print("  ✓ Passed")

    # Test 3: Cannot schedule when cooking
    print("\nTest 3: Cannot schedule when cooking")
    success, msg = cooker.update({"schedule_hours": 2})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    # Test 4: Complete cooking
    print("\nTest 4: Complete cooking")
    cooker.tick(1.0)
    assert cooker.status == "idle"
    assert cooker.is_busy() == False
    print("  ✓ Passed")

    # Test 5: CAN reschedule when in scheduled state (not cooking)
    print("\nTest 5: Can reschedule when in scheduled state")
    cooker2 = PressureCooker("Test Pressure Cooker 2")
    cooker2.update({"mode": "vegetables", "schedule_hours": 3})
    assert cooker2.status == "scheduled"
    success, msg = cooker2.update({"schedule_hours": 1})
    print(f"  Result: {msg}")
    assert success == True
    assert cooker2.status == "scheduled"
    assert cooker2.remaining_time == 1.0
    print("  ✓ Passed")


def test_pressure_cooker_reschedule():
    """Test PressureCooker rescheduling without mode"""
    print("\n=== Testing PressureCooker Reschedule ===")

    cooker = PressureCooker("Test Pressure Cooker")

    # Test 1: Schedule with mode
    print("Test 1: Schedule vegetables for 3 hours")
    success, msg = cooker.update({"mode": "vegetables", "schedule_hours": 3})
    print(f"  Result: {msg}")
    assert success == True
    assert cooker.status == "scheduled"
    assert cooker.pending_mode == "vegetables"
    assert "Scheduled" in msg
    print("  ✓ Passed")

    # Test 2: Reschedule without mode (uses pending_mode)
    print("\nTest 2: Reschedule to 1 hour without specifying mode")
    success, msg = cooker.update({"schedule_hours": 1})
    print(f"  Result: {msg}")
    assert success == True
    assert cooker.status == "scheduled"
    assert cooker.remaining_time == 1.0
    assert cooker.pending_mode == "vegetables"  # Should keep the same mode
    assert "Rescheduled" in msg  # Should say "Rescheduled"
    print("  ✓ Passed")

    # Test 3: Cancel schedule and start immediately
    print("\nTest 3: Cancel schedule, start immediately (schedule_hours=0)")
    success, msg = cooker.update({"schedule_hours": 0})
    print(f"  Result: {msg}")
    assert success == True
    assert cooker.status == "cooking"
    assert cooker.remaining_time == 0.5  # vegetables duration
    print("  ✓ Passed")


def test_pressure_cooker_full_flow():
    """Test PressureCooker complete flow: schedule -> cook -> complete"""
    print("\n=== Testing PressureCooker Full Flow ===")

    cooker = PressureCooker("Test Pressure Cooker")

    # Step 1: Schedule
    print("Step 1: Schedule beef_mutton for 1 hour delay")
    cooker.update({"mode": "beef_mutton", "schedule_hours": 1})
    assert cooker.status == "scheduled"
    assert cooker.remaining_time == 1.0
    assert cooker.pending_mode == "beef_mutton"
    assert cooker.pending_duration == 1.5
    print(f"  Status: {cooker.status}, remaining: {cooker.remaining_time}h")
    print("  ✓ Passed")

    # Step 2: Wait through schedule
    print("\nStep 2: Wait 1 hour (schedule completes)")
    cooker.tick(1.0)
    assert cooker.status == "cooking"
    assert cooker.remaining_time == 1.5  # Now cooking beef_mutton
    print(f"  Status: {cooker.status}, remaining: {cooker.remaining_time}h")
    print("  ✓ Passed")

    # Step 3: Cook for 1 hour
    print("\nStep 3: Cook for 1 hour")
    cooker.tick(1.0)
    assert cooker.status == "cooking"
    assert cooker.remaining_time == 0.5
    print(f"  Status: {cooker.status}, remaining: {cooker.remaining_time}h")
    print("  ✓ Passed")

    # Step 4: Complete cooking
    print("\nStep 4: Complete cooking (tick 0.5h)")
    cooker.tick(0.5)
    assert cooker.status == "idle"
    assert cooker.remaining_time == 0
    assert cooker.pending_mode is None
    assert cooker.is_busy() == False
    print(f"  Status: {cooker.status}")
    print("  ✓ Passed")


def test_pressure_cooker_broken():
    """Test PressureCooker in broken state"""
    print("\n=== Testing PressureCooker Broken State ===")

    cooker = PressureCooker("Test Pressure Cooker")
    cooker.break_device()

    print("Test 1: Read broken device")
    state = cooker.read()
    print(f"  Result: {state}")
    assert state.get("error") == "ConnectionError"
    print("  ✓ Passed")

    print("\nTest 2: Cannot cook when broken")
    success, msg = cooker.update({"mode": "vegetables"})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    print("\nTest 3: Cannot schedule when broken")
    success, msg = cooker.update({"schedule_hours": 1})
    print(f"  Result: {msg}")
    assert success == False
    print("  ✓ Passed")

    print("\nTest 4: Repair and cook")
    cooker.repair()
    success, msg = cooker.update({"mode": "vegetables"})
    print(f"  Result: {msg}")
    assert success == True
    print("  ✓ Passed")


if __name__ == "__main__":
    # Fridge tests
    test_fridge_basic()
    test_fridge_inventory_management()
    test_fridge_broken()

    # CoffeeMaker tests
    test_coffee_maker_basic()
    test_coffee_maker_recipes()
    test_coffee_maker_insufficient_ingredients()
    test_coffee_maker_broken()

    # PressureCooker tests
    test_pressure_cooker_basic()
    test_pressure_cooker_modes()
    test_pressure_cooker_schedule()
    test_pressure_cooker_schedule_validation()
    test_pressure_cooker_busy()
    test_pressure_cooker_reschedule()
    test_pressure_cooker_full_flow()
    test_pressure_cooker_broken()

    print("\n=== All Kitchen tests passed! ===")