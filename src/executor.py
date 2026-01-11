import sys
import json
import importlib.util  # Fixed: Must be explicitly imported for .util
import os
import re


def validate_path_safety(path):
    """Ensure no directory traversal and convert to absolute for orchestration stability."""
    if not path: return None
    if re.search(r'[<>:"|?*]', path):
        raise ValueError(f"Unsafe path detected: {path}")
    return os.path.abspath(path)


def load_manifest(manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if len(sys.argv) < 2:
    sys.exit(1)

manifest_path = validate_path_safety(sys.argv[1])
manifest = load_manifest(manifest_path)

provided_args = sys.argv[2:]
raw_cli_override = {provided_args[i].lstrip('-'): provided_args[i+1] 
                    for i in range(0, len(provided_args), 2) if i+1 < len(provided_args)}

input_map = {}
output_map = {}

mappings = sorted(manifest.get("input_arguments_mapping", []), key=lambda x: x.get('position', 0))

for mapping in mappings:
    label = mapping.get("label")
    m_type = mapping.get("type")
    # Priority: CLI Override -> Manifest disk_path
    path = validate_path_safety(raw_cli_override.get(label)) or validate_path_safety(mapping.get("disk_path"))

    if not path and not mapping.get("optional", False):
        print(f"Error: Required input '{label}' missing.")
        sys.exit(1)

    if path:
        if m_type == 'input_directory':
            if os.path.isdir(path):
                input_map[label] = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            else:
                input_map[label] = [] # Fallback for empty or missing dir
        elif m_type == 'input_file':
            input_map[label] = path
        elif m_type == 'script_file':
            # DYNAMIC IMPORT: Enable 'import label'
            script_dir = os.path.dirname(path)
            module_name = os.path.splitext(os.path.basename(path))[0]
            
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            
            # Load module and inject into globals under the Label name
            spec = importlib.util.spec_from_file_location(module_name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            globals()[label] = mod # Key: script is now accessible by its LABEL
            input_map[label] = path

# --- 2. Resolve OUTPUTS (Dir & File) ---

out_mappings = sorted(manifest.get("outputs_definition_mapping", []), key=lambda x: x.get('position', 0))
for out_map in out_mappings:
    label = out_map.get("label")
    m_type = out_map.get("type")
    path = validate_path_safety(raw_cli_override.get(label)) or validate_path_safety(out_map.get("disk_path"))

    if not path:
        print(f"Error: Output '{label}' missing destination.")
        sys.exit(1)

    if m_type == "output_dir":
        os.makedirs(path, exist_ok=True)
    elif m_type == "output_file":
        # Ensure the parent directory exists for the output file
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    output_map[label] = path

# --- 3. Execute Entry Point .process ---
entry_path = validate_path_safety(manifest.get("entry_point"))
if not entry_path or not os.path.exists(entry_path):
    print(f"Error: Entry point script missing at {entry_path}")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("entry_point", entry_path)
entry_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry_mod)

try:
    receipt = entry_mod.process(input_map, output_map)
    
    # Write result for C# Orchestrator
    with open("process_receipt.json", "w", encoding='utf-8') as f:
        json.dump(receipt, f, indent=4)
except Exception as e:
    # Record crash in the receipt
    with open("process_receipt.json", "w", encoding='utf-8') as f:
        json.dump({"status": "error", "message": str(e)}, f)
    sys.exit(1)