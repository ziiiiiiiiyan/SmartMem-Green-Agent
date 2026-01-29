"""
Smart Home Device Operations Configuration
Lists all devices, their operable parameters, and all possible values
"""

DEVICE_OPERATIONS = {
    "AC": {
        "actions": ["read", "update"],
        "update_params": {
            "power": {
                "values": ["on", "off"],
                "type": "str"
            },
            "mode": {
                "values": ["cooling", "heating", "dehumidify"],
                "type": "str"
            },
            "fan_speed": {
                "values": ["auto", "1", "2", "3"],
                "type": "str"
            },
            "sleep_mode": {
                "values": ["on", "off"],
                "type": "str"
            },
            "temperature": {
                "values": list(range(16, 31)),  # 16-30
                "type": "int"
            },
            "timer": {
                "values": [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                "type": "float",
                "constraints": "0-5, multiples of 0.5"
            }
        }
    },
    "Bedroom Light": {
        "actions": ["read", "update"],
        "update_params": {
            "power": {
                "values": ["on", "off"],
                "type": "str"
            },
            "color": {
                "values": ["white", "red", "blue", "warm"],
                "type": "str",
                "note": "This light supports color changes"
            }
        }
    },
    "Living Room Light": {
        "actions": ["read", "update"],
        "update_params": {
            "power": {
                "values": ["on", "off"],
                "type": "str"
            },
            "color": {
                "values": ["white", "red", "blue", "warm"],
                "type": "str",
                "note": "This light supports color changes"
            }
        }
    },
    "Kitchen Light": {
        "actions": ["read", "update"],
        "update_params": {
            "power": {
                "values": ["on", "off"],
                "type": "str"
            },
            "color": {
                "values": None,
                "type": "str",
                "note": "This light does NOT support color changes"
            }
        }
    },
    "Speaker": {
        "actions": ["read", "update"],
        "update_params": {
            "volume": {
                "values": list(range(11)),  # 0-10
                "type": "int"
            }
        }
    },
    "Security": {
        "actions": ["read", "update"],
        "update_params": {
            "door_lock": {
                "values": ["open", "closed"],
                "type": "str"
            },
            "intercom_reply": {
                "values": "any_string",
                "type": "str",
                "note": "Can be any string message"
            }
        }
    },
    "Washing Machine": {
        "actions": ["read", "update"],
        "update_params": {
            "mode": {
                "values": ["quick", "mix", "wool", "dry_1h", "dry_2h", "dry_3h", "dry"],
                "type": "str",
                "note": "If using 'dry', must also provide 'dry_hours' parameter"
            },
            "dry_hours": {
                "values": [1, 2, 3],
                "type": "int",
                "note": "Required only when mode='dry'"
            },
            "schedule_hours": {
                "values": list(range(1, 11)),  # 1-10
                "type": "int",
                "note": "Requires mode to be set first"
            }
        }
    },
    "Fridge": {
        "actions": ["read"],
        "update_params": {},
        "note": "Fridge only supports read action to check inventory"
    },
    "Coffee Maker": {
        "actions": ["read", "update"],
        "update_params": {
            "make_coffee": {
                "values": ["americano", "latte", "oat_latte", "mocha"],
                "type": "str",
                "note": "Requires sufficient ingredients in fridge"
            }
        }
    },
    "Pressure Cooker": {
        "actions": ["read", "update"],
        "update_params": {
            "mode": {
                "values": ["beef_mutton", "chicken_duck", "vegetables"],
                "type": "str"
            },
            "schedule_hours": {
                "values": [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
                "type": "float",
                "constraints": "0-8, multiples of 0.5",
                "note": "0 means start immediately"
            }
        }
    }
}


def print_device_operations():
    """Print device operations in a readable format"""
    import json
    print(json.dumps(DEVICE_OPERATIONS, indent=2, ensure_ascii=False))


def get_device_params(device_name):
    """Get parameters for a specific device"""
    return DEVICE_OPERATIONS.get(device_name, {})


def generate_example_command(device_name, action="update", **kwargs):
    """
    Generate an example command for a device

    Example:
        generate_example_command("AC", power="on", mode="cooling", timer=2)
        Returns: {"device": "AC", "action": "update", "params": {"power": "on", "mode": "cooling", "timer": 2}}
    """
    device_config = DEVICE_OPERATIONS.get(device_name)
    if not device_config:
        raise ValueError(f"Unknown device: {device_name}")

    if action not in device_config["actions"]:
        raise ValueError(f"Device {device_name} does not support action '{action}'")

    return {
        "device": device_name,
        "action": action,
        "params": kwargs if action == "update" else {}
    }


if __name__ == "__main__":
    print_device_operations()

    # Example usage
    print("\n\n" + "=" * 80)
    print("EXAMPLE COMMANDS")
    print("=" * 80)

    examples = [
        generate_example_command("AC", mode="cooling", fan_speed="auto", timer=2),
        generate_example_command("Bedroom Light", color="blue"),
        generate_example_command("Speaker", volume=8),
        generate_example_command("Washing Machine", mode="quick", schedule_hours=3),
        generate_example_command("Coffee Maker", make_coffee="latte"),
        generate_example_command("Pressure Cooker", mode="beef_mutton", schedule_hours=2.0),
    ]

    for ex in examples:
        print(f"\n{ex}")
