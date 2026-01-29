from .devices import Fridge, Light, AirConditioner, Speaker, SecuritySystem, WashingMachine, CoffeeMaker, PressureCooker
from .utils import print_table


class SmartHomeEnv:
    def __init__(self):
        self.history = []
        self.time_simulation_elapsed = 0.0
        
        # Instantiate Devices
        self.fridge = Fridge("Fridge")
        self.devices = {
            "Bedroom Light": Light("Bedroom Light", has_color=True),
            "Living Room Light": Light("Living Room Light", has_color=True),
            "Kitchen Light": Light("Kitchen Light", has_color=False),
            "AC": AirConditioner("AC"),
            "Speaker": Speaker("Speaker"),
            "Security": SecuritySystem("Security"),
            "Washing Machine": WashingMachine("Washing Machine"),
            "Fridge": self.fridge,
            "Coffee Maker": CoffeeMaker("Coffee Maker", self.fridge),
            "Pressure Cooker": PressureCooker("Pressure Cooker")
        }
        

    def tick(self, hours):
        """Advances time by specified hours."""
        self.time_simulation_elapsed += hours
        for device in self.devices.values():
            device.tick(hours)

    def get_status_report(self):
        """Generates a text table of all devices."""
        headers = ["Device", "Status", "Details / Config"]
        rows = []
        for name, device in self.devices.items():
            state = device.read()
            broken_str = "[BROKEN] " if state["broken"] else ""
            
            # Format specific details based on device type for readability
            details = []
            for k, v in state.items():
                if k not in ["name", "broken"]:
                    details.append(f"{k}:{v}")
            
            detail_str = ", ".join(details)
            rows.append([name, broken_str + ("Busy" if device.is_busy() else "Idle/Active"), detail_str])

        return print_table(headers, rows)

    def set_state(self, state_dict):
        """
        Sets state for multiple devices at once.
        Input: {"Bedroom Light": {"power": "on"}, ...}
        """
        results = {}
        allsuccess = True
        for dev_name, params in state_dict.items():
            if dev_name in self.devices:
                success, msg = self.devices[dev_name].update(params)
                results[dev_name] = "Success" if success else f"Failed: {msg}"
                if not results[dev_name] == 'Success':
                    allsuccess = False
            else:
                results[dev_name] = "Device not found"
        return allsuccess, results

    def execute(self, command):
        """
        Unified Interface.
        Input: {"device": str, "action": "read"|"update", "params": dict}
        Output: JSON dict
        """
        dev_name = command.get("device")
        action = command.get("action")
        params = command.get("params", {})

        # Validation
        if dev_name not in self.devices:
            return {"device": dev_name, "status": False, "message": "Device not found."}
        
        device = self.devices[dev_name]
        
        # Execute Action
        result = {}
        if action == "read":
            data = device.read()
            result = {"device": dev_name, "status": True, "message": "Read success", "data": data}
        elif action == "update":
            success, msg = device.update(params)
            result = {"device": dev_name, "status": success, "message": msg}
        else:
            return {"device": dev_name, "status": False, "message": "Unknown action."}

        # Log History
        self.history.append({
            "command": command,
            "result": result,
            "global_time_elapsed": self.time_simulation_elapsed
        })

        return result

    # --- Accident System ---

    def trigger_accident(self, target_name):
        """
        Deterministically triggers a specific event (Device Break or Food Depletion).
        
        Args:
            target_name (str): The specific name of the device (e.g., "AirConditioner") 
                               or the food item (e.g., "beef").
        
        Returns:
            dict: {
                "success": bool,
                "type": "device_error" | "food_depletion" | "error",
                "target": str,
                "detail": str  # The message to be pushed to the Agent
            }
        """
        if not target_name:
            return {
                "success": False,
                "type": "error",
                "target": "None",
                "detail": "No target specified for event trigger."
            }

        # --- 1. 检查是否为设备 (Device Logic) ---
        if target_name in self.devices:
            device = self.devices[target_name]
            
            # 即使设备已经坏了，再次触发也算成功（状态符合预期）
            # 这里主要是模拟连接丢失
            device.break_device()
            
            return {
                "success": True,
                "type": "device_error",
                "target": target_name,
                "detail": f"[ALERT] System Monitor: Lost connection to '{target_name}'. Device is unresponsive."
            }

        # --- 2. 检查是否为食材 (Food Logic) ---
        # 假设 fridge.inventory 是一个字典 { "beef": 2, "milk": 1 ... }
        if target_name in self.fridge.inventory:
            # 依赖检查：如果冰箱坏了（传感器离线），则无法感知库存变化
            if self.fridge.is_broken:
                return {
                    "success": False,
                    "type": "food_depletion",
                    "target": target_name,
                    "detail": "Fridge sensor is offline. Cannot update inventory status."
                }
            
            # 执行：库存清零（模拟被消耗完）
            self.fridge.set_inventory(target_name, 0)
            
            return {
                "success": True,
                "type": "food_depletion",
                "target": target_name,
                "detail": f"[SMART FRIDGE] Alert: '{target_name}' is out of stock. Please restock."
            }

        # --- 3. 目标不存在 ---
        return {
            "success": False,
            "type": "error",
            "target": target_name,
            "detail": f"Target '{target_name}' not found in system."
        }

    def repair_system(self, target_device_name=None):
        """
        Fixes specific device or all devices.
        """
        if target_device_name:
            if target_device_name in self.devices:
                dev = self.devices[target_device_name]
                if dev.is_broken:
                    dev.repair()
                    return {"type": "repair", "target": target_device_name, "status": "Fixed"}
                elif target_device_name == "Fridge":
                    # Determine if we are fixing the device or restocking? 
                    # Prompt says "restore values according to input" -> implies manual update.
                    # But "repair" interface usually fixes the 'broken' status.
                    # Let's assume this fixes the mechanical 'is_broken' status.
                    return {"type": "repair", "target": target_device_name, "status": "Device is now fixed."}
            return {"type": "repair", "target": target_device_name, "status": "Device not found"}
        
        else:
            # Fix ALL
            repaired_list = []
            for name, dev in self.devices.items():
                if dev.is_broken:
                    dev.repair()
                    repaired_list.append(name)
            return {"type": "repair", "target": "ALL", "repaired_devices": repaired_list}