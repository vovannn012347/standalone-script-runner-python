import sys
import json
import importlib.util
import os
import argparse
import re


def make_path_safe(path: str) -> str | None:
    """Prevents directory traversal and normalizes separators to Linux-style."""
    if not path:
        return None
    # Normalize Windows backslashes → forward slashes
    path = path.replace("\\", "/")
    # Reject invalid / dangerous characters
    if re.search(r'[<>:"|?*]', path):
        raise ValueError(f"Unsafe path characters detected: {path}")
    # Normalize path (., .., duplicate slashes)
    path = os.path.normpath(path)
    return path


def validate_path(path, base_dir=None):
    """Resolves and normalizes paths relative to a provided base directory."""
    if not path: return None
    safe_path = make_path_safe(path)
    if not os.path.isabs(safe_path) and base_dir:
        # Join using the provided base directory
        safe_path = os.path.join(base_dir, safe_path)
    # Ensure final path uses forward slashes and is absolute
    return os.path.normpath(os.path.abspath(safe_path)).replace("\\", "/")


def load_json(path):
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


class JsonCache:
    """Caches JSON file contents for fast retrieval during input resolution."""
    def __init__(self):
        self._cache = {}

    def get(self, path):
        if not path or not os.path.exists(path):
            return {}
        norm_path = os.path.normpath(path)
        if norm_path not in self._cache:
            with open(norm_path, 'r', encoding='utf-8') as f:
                self._cache[norm_path] = json.load(f)
        return self._cache[norm_path]

    def clear(self):
        """Discards cached data to free up memory for medical processing."""
        self._cache.clear()


def main():
    parser = argparse.ArgumentParser(description="Medical Script Standalone Executor")
    parser.add_argument("--processing-dir", required=True, help="Path to the input/output workspace")
    parser.add_argument("--script-source-dir", required=True, help="Path to the source script workspace")
    parser.add_argument("--manifest-file", required=True,
                        help="Path to the current processing manifest (relative to processing-dir)")
    args = parser.parse_args()

    # 1. Setup Contexts
    proc_dir = os.path.abspath(args.processing_dir).replace("\\", "/")
    script_root = os.path.abspath(args.script_source_dir).replace("\\", "/")
    json_loader = JsonCache()  # Initialize cache

    # 2. Load Step Manifest
    manifest_path = validate_path(args.manifest_file, proc_dir)
    proc_manifest = json_loader.get(manifest_path)
    if not proc_manifest:
        print(f"Error: Manifest file not found at {manifest_path}")
        sys.exit(1)

    # Determine Output target based on folder_base
    folder_base_rel = proc_manifest.get("folder_base", ".")
    output_target_dir = validate_path(folder_base_rel, proc_dir)
    os.makedirs(output_target_dir, exist_ok=True)

    # 3. Load Script Metadata
    script_manifest = json_loader.get(os.path.join(script_root, "script_manifest.json"))
    entry_point_name = script_manifest.get("entry_point")
    entry_point_path = validate_path(entry_point_name, script_root)

    input_map = {"__processing_dir__": proc_dir}
    output_map = {}
    proc_inputs_base = proc_manifest.get("inputs_base", {})

    # 4. Resolve Inputs with Caching
    for mapping in script_manifest.get("inputs_mapping", []):
        label = mapping.get("label")
        m_type = mapping.get("type")
        disk_path = mapping.get("disk_path")

        # Scalar Types: Extract from the specific JSON path provided in inputs_base
        if m_type in ["boolean", "integer", "decimal", "string"]:
            file_rel_path = proc_inputs_base.get(label)
            if file_rel_path:
                scalar_file_path = validate_path(file_rel_path, proc_dir)
                # Cache-enabled load
                scalar_data = json_loader.get(scalar_file_path)
                input_map[label] = scalar_data.get(label, mapping.get("default"))
            else:
                input_map[label] = mapping.get("default")
            continue

        # Read-Only Source Files (Resolved against script source)
        if m_type in ["source_file", "source_directory", "script_file"]:
            resolved = validate_path(disk_path, script_root)
            if m_type == "script_file":
                spec = importlib.util.spec_from_file_location(label, resolved)
                mod = importlib.util.module_from_spec(spec)
                sys.path.insert(0, os.path.dirname(resolved))
                spec.loader.exec_module(mod)
                input_map[label] = mod
            else:
                input_map[label] = resolved
            continue

        # User/Workspace Data (Resolved against proc_dir via inputs_base)
        rel_base_path = proc_inputs_base.get(label, ".")
        resolved_base = validate_path(rel_base_path, proc_dir)
        resolved_path = validate_path(disk_path if disk_path else "", resolved_base)

        if m_type == "folder":
            if os.path.isdir(resolved_path):
                input_map[label] = [os.path.join(resolved_path, f).replace("\\", "/")
                                    for f in os.listdir(resolved_path)
                                    if os.path.isfile(os.path.join(resolved_path, f))]
            else:
                input_map[label] = []
        else:
            input_map[label] = resolved_path

    # --- Optimization: Discard cache before execution to free memory ---
    json_loader.clear()

    # 5. Resolve Outputs
    proc_outputs_base = proc_manifest.get("outputs_base", {})
    for out_mapping in script_manifest.get("outputs_mapping", []):
        label = out_mapping.get("label")
        m_type = out_mapping.get("type")
        disk_path = out_mapping.get("disk_path", "")

        if m_type in ["integer", "decimal", "boolean", "string"]:
            continue

        out_rel_base = proc_outputs_base.get(label, ".")
        resolved_base = validate_path(out_rel_base, proc_dir)
        resolved_out = validate_path(disk_path, resolved_base)

        if m_type == "folder":
            os.makedirs(resolved_out, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(resolved_out), exist_ok=True)
        output_map[label] = resolved_out

    # 6. Execute Medical Script
    spec = importlib.util.spec_from_file_location("main_entry", entry_point_path)
    entry_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry_mod)

    try:
        # returns (summary, direct_outputs, file_outputs)
        summary, direct_out, file_out = entry_mod.process(input_map, output_map)

        # 7. Persistence (Targeting folder_base)
        with open(os.path.join(output_target_dir, "direct_output.json"), "w", encoding='utf-8') as f:
            json.dump(direct_out, f, indent=4)
        with open(os.path.join(output_target_dir, "file_output.json"), "w", encoding='utf-8') as f:
            json.dump(file_out, f, indent=4)
        with open(os.path.join(output_target_dir, "script_summary.json"), "w", encoding='utf-8') as f:
            json.dump(summary, f, indent=4)

    except Exception as e:
        with open(os.path.join(output_target_dir, "script_summary.json"), "w", encoding='utf-8') as f:
            json.dump({"status": "error", "message": str(e)}, f, indent=4)
        sys.exit(1)


if __name__ == "__main__":
    main()
