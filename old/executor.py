import sys
import json
import importlib.util
import os

# === Args: script, model, input, output ===
if len(sys.argv) != 5:
    print("Running in standalone mode...")
    sys.exit(1)
script_path, model_path, input_path, output_path = sys.argv[1:5]

# === Load & run user script ===
spec = importlib.util.spec_from_file_location("user", script_path)
user = importlib.util.module_from_spec(spec)
spec.loader.exec_module(user)

try:
    result = user.process(model_path, input_path)
    with open(output_path, "w") as f:
        json.dump(result, f)
except Exception as e:
    with open(output_path, "w") as f:
        json.dump({"error": str(e)}, f)
    sys.exit(1)