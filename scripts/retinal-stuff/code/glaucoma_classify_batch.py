import os
import json
import retina_nerve_definitor
import retina_nerve_glaucoma_classifier


def check_inputs(input_map, output_map):
    # no errors, this stuff runs even if there are no input files or no inputs,
    # will output default result
    return True, ""


def process(input_map, output_map):
    input_images = input_map.get("retinal_images", [])
    mask_images_dir = output_map.get("retinal_masks_images")
    result_json = output_map.get("retinal_classify_result")

    # Initialize models
    loaded_definitor, device = retina_nerve_definitor.load_definitor(input_map.get("definitor_model"))
    loaded_classifier, device = retina_nerve_glaucoma_classifier.load_classifier(input_map.get("classifier_model"))

    global_summary = {
        "status": "success",
        "message": "Batch complete",
        "errors": []
    }
    batch_results = []

    batch_counts = {
        "suspected_glaucoma": 0,
        "glaucoma": 0,
        "suspected_atrophy": 0,
        "atrophy": 0
    }
    file_outputs = {"nerve_crop" : []}
    nerve_cropper_file_outputs = []
    file_outputs["nerve_crop"] = nerve_cropper_file_outputs

    for index, img_path in enumerate(input_images):
        image_record = {
            "nerve_definition": [],
            "glaucoma_detection": [],
            "summary": []
        }

        rel_crops = []
        try:
            crop_paths = retina_nerve_definitor.run_definitor(
                image_path=img_path,
                output_path=os.path.join(mask_images_dir, str(index)),
                loaded_definitor=loaded_definitor,
                premade_device=device
            )

            rel_crops = [os.path.relpath(c, mask_images_dir) for c in crop_paths]
            image_record["nerve_definition"].append({
                "input_files": [os.path.basename(img_path)],
                "output_files": rel_crops
            })

            nerve_cropper_file_outputs.extend(rel_crops)
        except Exception as e:
            global_summary["errors"].append({"file": img_path, "error": str(e)})

            # 2. Glaucoma Detection Process (Iterative per crop)
        valid_crop_scores = []
        for crop_rel in rel_crops:
            try:
                full_path = os.path.join(mask_images_dir, crop_rel)
                scores = retina_nerve_glaucoma_classifier.run_classifier(full_path, loaded_classifier, device)

                image_record["glaucoma_detection"].append({
                    "input_files": [crop_rel],
                    "output": scores
                })

                if scores.get("valid_image", 0) > 0.5:
                    valid_crop_scores.append(scores)
            except Exception as e:
                global_summary["errors"].append({"file": crop_rel, "error": str(e)})

        labels = ["glaucoma", "atrophy", "valid_image"]
        final_outputs = {label: max(max([s[label] for s in valid_crop_scores]), 0) for label in labels}

        g_score = final_outputs["glaucoma"]
        a_score = final_outputs["atrophy"]

        if g_score >= 0.8:
            batch_counts["glaucoma"] += 1
        elif g_score >= 0.4:
            batch_counts["suspected_glaucoma"] += 1

        if a_score >= 0.8:
            batch_counts["atrophy"] += 1
        elif a_score >= 0.5:
            batch_counts["suspected_atrophy"] += 1

        image_record["summary"].append({
            "input": valid_crop_scores,  # Diagnostic thought process
            "output": final_outputs
        })

        batch_results.append(image_record)

    os.makedirs(os.path.dirname(result_json), exist_ok=True)
    with open(result_json, "w", encoding='utf-8') as f:
        json.dump(batch_results, f, indent=4)

    summary_parts = []
    for key, count in batch_counts.items():
        if count > 0:
            summary_parts.append(f"{key}: {count}")

    if not summary_parts:
        global_summary["message"] = "normal_retina"
    else:
        global_summary["message"] = ", ".join(summary_parts)

    if global_summary["errors"]:
        global_summary["status"] = "errors"

    return global_summary, {}, file_outputs

