import importlib

# List of module names to import
module_names = ["module1", "module2", "module3"]

# Loop through the module names and import them
for module_name in module_names:
    try:
        module = importlib.import_module(module_name)
        # You can now use functions or classes from the imported module
        # For example: module.some_function()
    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")
