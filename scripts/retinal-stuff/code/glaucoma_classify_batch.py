import os
import json
import retina_nerve_definitor
import retina_nerve_glaucoma_classifier


def process(input_map, output_map):
    # 1. Retrieve Absolute Paths from Executor
    input_images = input_map.get("retinal_images", [])
    mask_images_dir = output_map.get("retinal_masks_images")
    result_json = output_map.get("retinal_classify_result")

    # We need the processing_dir root to calculate relative paths for the orchestrator
    # The executor should pass this as a special entry in input_map
    proc_root = input_map.get("__processing_dir__", os.getcwd())

    # 2. Initialize Medical Models using weights from input_map
    loaded_definitor, device = retina_nerve_definitor.load_definitor(input_map.get("definitor_model"))
    loaded_classifier, device = retina_nerve_glaucoma_classifier.load_classifier(input_map.get("classifier_model"))

    global_summary = {"status": "success", "message": "Batch complete", "errors": []}
    batch_results = []

    # Track counts for the 'direct_outputs' manifest
    batch_counts = {
        "glaucoma": 0, "suspected_glaucoma": 0,
        "atrophy": 0, "suspected_atrophy": 0
    }

    # Track file paths for 'file_outputs' manifest
    generated_crops = []

    for index, img_path in enumerate(input_images):
        image_record = {"nerve_definition": [], "glaucoma_detection": [], "summary": []}

        try:
            # 3. Nerve Definition (Crop Generation)
            crop_paths = retina_nerve_definitor.run_definitor(
                image_path=img_path,
                output_path=os.path.join(mask_images_dir, str(index)),
                loaded_definitor=loaded_definitor,
                premade_device=device
            )

            # Convert to relative paths for the orchestrator
            rel_crops = [os.path.relpath(c, proc_root).replace("\\", "/") for c in crop_paths]
            generated_crops.extend(rel_crops)

            image_record["nerve_definition"].append({
                "input_files": [os.path.basename(img_path)],
                "output_files": rel_crops
            })

            # 4. Glaucoma Classification
            valid_crop_scores = []
            for crop_abs_path in crop_paths:
                try:
                    scores = retina_nerve_glaucoma_classifier.run_classifier(crop_abs_path, loaded_classifier, device)

                    if scores.get("valid_image", 0) > 0.5:
                        valid_crop_scores.append(scores)

                        # Aggregated Count Logic
                        if scores["glaucoma"] >= 0.8:
                            batch_counts["glaucoma"] += 1
                        elif scores["glaucoma"] >= 0.4:
                            batch_counts["suspected_glaucoma"] += 1

                        if scores["atrophy"] >= 0.8:
                            batch_counts["atrophy"] += 1
                        elif scores["atrophy"] >= 0.5:
                            batch_counts["suspected_atrophy"] += 1

                except Exception as e:
                    global_summary["errors"].append({"file": os.path.basename(crop_abs_path), "error": str(e)})

            # 5. Build Final Image Summary
            labels = ["glaucoma", "atrophy", "valid_image"]
            final_scores = {label: max([s[label] for s in valid_crop_scores]) if valid_crop_scores else 0 for label in
                            labels}

            image_record["summary"].append({"output": final_scores})
            batch_results.append(image_record)

        except Exception as e:
            global_summary["errors"].append({"file": os.path.basename(img_path), "error": str(e)})

    # 6. Persist Detailed Batch JSON
    os.makedirs(os.path.dirname(result_json), exist_ok=True)
    with open(result_json, "w", encoding='utf-8') as f:
        json.dump(batch_results, f, indent=4)

    # 7. Map Outputs to Manifest Labels
    direct_outputs = {
        "retinal_glaucoma_count": batch_counts["glaucoma"],
        "retinal_glaucoma_probability_0_1": max(
            [r["summary"][0]["output"]["glaucoma"] for r in batch_results]) if batch_results else 0.0,
        "retinal_atrophy_count": batch_counts["atrophy"],
        "retinal_atrophy_probability_0_1": max(
            [r["summary"][0]["output"]["atrophy"] for r in batch_results]) if batch_results else 0.0
    }

    file_outputs = {
        "retinal_masks_images": generated_crops  # List of relative paths
    }

    return global_summary, direct_outputs, file_outputs