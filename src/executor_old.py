import sys
import json
import importlib.util
import os
import argparse
import re


def validate_path_safety(path):
    """Ensure no directory traversal and convert to absolute for orchestration stability."""
    if not path: return None
    if re.search(r'[<>:"|?*]', path):
        raise ValueError(f"Unsafe path detected: {path}")
    return os.path.abspath(path)


def validate_path(path, base_dir=None):
    """Joins relative paths with a base directory and ensures absolute paths."""
    if not path:
        return None
    if not os.path.isabs(path) and base_dir:
        path = os.path.join(base_dir, path)
    return os.path.abspath(path)


def load_json(path):
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Standalone Script Executor")
    parser.add_argument("--processing-dir", required=True, help="Task directory containing manifest and data")
    args = parser.parse_args()

    proc_dir = os.path.abspath(args.processing_dir)

    # 1. Load processing artifacts from processing-dir
    # processing_manifest considered safe, is constructed by the processing system
    proc_manifest = load_json(os.path.join(proc_dir, "orchestration_manifest.json"))
    # considered safe, no file path should be in here
    direct_inputs = load_json(os.path.join(proc_dir, "direct_input.json"))

    # 2. Locate and load script_manifest
    script_root = proc_manifest.get("run-script")
    if not script_root:
        print("Error: 'run-script' not defined in orchestration_manifest.json")
        sys.exit(1)

    # script_manifest paths are considered unsafe
    script_manifest = load_json(os.path.join(script_root, "script_manifest.json"))
    entry_point_name = script_manifest.get("entry_point")
    entry_point_path = validate_path(entry_point_name, script_root)

    input_map = {}
    output_map = {}

    input_map.update(direct_inputs)

    # 3. Resolve Inputs
    proc_inputs_base = proc_manifest.get("inputs_base", {})
    for mapping in script_manifest.get("inputs_mapping", []):
        label = mapping.get("label")
        m_type = mapping.get("type")
        path_relative_end = mapping.get("disk_path")

        if not path_relative_end and not (mapping.get("optional", False) or proc_inputs_base[label]):
            print(f"Error: Required input '{label}' missing.")
            sys.exit(1)

        validate_path_safety(path_relative_end)

        # source_file, source_directory, script_file are not allowed
        # to be overridden by inputs if path is provided

        # if we have path_relative on file - means it is non-user input file
        if m_type == "source_file":
            if path_relative_end:
                resolved_path = validate_path(path_relative_end, script_root)
            else:
                resolved_path = proc_inputs_base[label]
            input_map[label] = resolved_path
            continue

        if m_type == "source_directory":
            if path_relative_end:
                resolved_path = validate_path(path_relative_end, script_root)
            else:
                resolved_path = proc_inputs_base[label]

            if resolved_path and os.path.isdir(resolved_path):
                input_map[label] = [os.path.join(resolved_path, f) for f in os.listdir(resolved_path) if
                                    os.path.isfile(os.path.join(resolved_path, f))]
            else:
                input_map[label] = []
            continue

        if m_type == "script_file":
            if path_relative_end:
                mod_path = validate_path(path_relative_end, script_root)
            else:
                mod_path = proc_inputs_base[label]
            spec = importlib.util.spec_from_file_location(label, mod_path)
            mod = importlib.util.module_from_spec(spec)
            sys.path.insert(0, os.path.dirname(mod_path))
            spec.loader.exec_module(mod)
            globals()[label] = mod
            continue

        if path_relative_end:
            path_relative_start = proc_inputs_base[label]
            if path_relative_start:
                resolved_path = validate_path(path_relative_end,  path_relative_start)
            else:
                resolved_path = validate_path(path_relative_end,  proc_dir)
        else:
            resolved_path = proc_inputs_base[label]

        if m_type == "folder":
            if resolved_path and os.path.isdir(resolved_path):
                input_map[label] = [os.path.join(resolved_path, f) for f in os.listdir(resolved_path) if
                                    os.path.isfile(os.path.join(resolved_path, f))]
            else:
                input_map[label] = []
            continue

        if m_type == "file":
            input_map[label] = resolved_path
            continue

    # 4. Resolve Outputs
    proc_outputs_base = proc_manifest.get("outputs_base", {})
    for out_mapping in script_manifest.get("outputs_mapping", []):
        label = out_mapping.get("label")
        m_type = out_mapping.get("type")
        path_relative_end = out_mapping.get("disk_path")

        # Ensure the path string itself doesn't contain illegal characters
        validate_path_safety(path_relative_end)

        # 1. Determine the Base Resolution Path
        # Priority: proc_outputs mapping -> processing-dir
        path_relative_start = proc_outputs_base.get(label)

        if path_relative_end:
            if path_relative_start:
                resolved_path = validate_path(path_relative_end, path_relative_start)
            else:
                resolved_path = validate_path(path_relative_end, proc_dir)
        else:
            resolved_path = path_relative_start

        # 2. Handle Directory/File Creation
        if m_type == "folder":
            if resolved_path:
                # Create the directory if it does not exist
                os.makedirs(resolved_path, exist_ok=True)

                # Since this is an output folder mapping, we provide the path itself
                # to the script so it knows where to write files.
                output_map[label] = resolved_path
            else:
                output_map[label] = None
            continue

        if m_type == "file":
            if resolved_path:
                # Ensure the parent directory for the output file exists
                parent_dir = os.path.dirname(resolved_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)

                output_map[label] = resolved_path
            else:
                output_map[label] = None
            continue

    # 5. Execute
    spec = importlib.util.spec_from_file_location("main_entry", entry_point_path)
    entry_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry_mod)

    try:
        process_result, direct_output, file_output = entry_mod.process(input_map, output_map)

        output_result_path = os.path.join(proc_dir, "direct_output.json")
        output_file_result_path = os.path.join(proc_dir, "file_output.json")

        with open(output_result_path, "w", encoding='utf-8') as f:
            json.dump(direct_output, f, indent=4)

        with open(output_file_result_path, "w", encoding='utf-8') as f:
            json.dump(file_output, f, indent=4)

        with open(os.path.join(proc_dir, "script_summary.json"), "w", encoding='utf-8') as f:
            json.dump(process_result, f, indent=4)

    except Exception as e:
        with open(os.path.join(proc_dir, "script_summary.json"), "w", encoding='utf-8') as f:
            json.dump({"status": "error", "message": str(e)}, f, indent=4)
        sys.exit(1)


if __name__ == "__main__":
    main()