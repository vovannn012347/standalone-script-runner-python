import os
import json
import retina_nerve_definitor
import retina_nerve_glaucoma_classifier


def check_inputs(input_map, output_map):
    # todo: add input check with explainable errors

    return True, ""


def process(input_map, output_map):

    """
        Orchestrates the retinal nerve masking and stripping process.

        1) Loads the definitor model using pretrained weights.
        2) Iterates through input images one by one.
        3) Saves cropped optic nerve images to the output folder.
        4) Generates a summary receipt.
        """

    definitor_weights_path = input_map.get("definitor_model")
    classifier_weights_path = input_map.get("classifier_model")

    input_image_paths = input_map.get("retinal_images", [])

    # Output paths
    mask_images_dir = output_map.get("retinal_masks_images")
    mask_classify_info_dir = output_map.get("retinal_masks_info")
    result_json_path = output_map.get("retinal_classify_result")

    # 2. Initialize the model
    # use_cuda logic is handled inside load_definitor
    loaded_definitor, device = retina_nerve_definitor.load_definitor(definitor_weights_path)
    loaded_classifier, device = retina_nerve_glaucoma_classifier.load_classifier(classifier_weights_path)

    direct_result = {}
    file_result = {}
    processing_summary = {
        "status": "error",
        "message": "",
        "errors": []
    }
    nerve_definition_results = []
    file_result["retinal_masks_images"] = nerve_definition_results

    # 3. Process images one by one
    for index, img_path in enumerate(input_image_paths):
        try:
            # run_definitor handles: pre-processing, 0.5 interpolation,
            # magic wand selection, and saving the crops.
            generated_crops = retina_nerve_definitor.run_definitor(
                image_path=img_path,
                output_path=os.path.join(mask_images_dir, str(index)),
                loaded_definitor=loaded_definitor,
                premade_device=device
            )

            relative_crops = [
                os.path.relpath(crop_path, mask_images_dir)
                for crop_path in generated_crops
            ]

            # 3. Append cleaned results
            nerve_definition_results.append({
                "source_file": os.path.basename(img_path),
                "out_files": relative_crops
            })

        except Exception as e:
            processing_summary["errors"].append({
                "source_file": os.path.basename(img_path),
                "error": str(e)
            })

    # process detected nerve results one by one
    nerve_classify_results = []
    file_result["retinal_masks_info"] = nerve_definition_results

    per_file_results = []
    for index_definition_result, definition_result in enumerate(nerve_definition_results):
        file_results = {
            "result": {},
            "subfiles": []
        }
        for index_definition_sub_result, classify_file in enumerate(definition_result["out_files"]):
            try:
                # classifier handles: pre-processing
                classify_file_path = os.path.join(mask_images_dir, classify_file)
                classify_results = retina_nerve_glaucoma_classifier.run_classifier(
                    image_path=classify_file_path,
                    loaded_classifier=loaded_classifier,
                    premade_device=device
                )

                classify_result_file_path = os.path.join(mask_classify_info_dir, classify_file) + ".json"
                os.makedirs(os.path.dirname(classify_result_file_path), exist_ok=True)

                with open(classify_result_file_path, "w", encoding='utf-8') as f:
                    json.dump(classify_results, f, indent=4)

                file_results["subfiles"].append(classify_result_file_path)

            except Exception as e:
                processing_summary["errors"].append({
                    "source_file": os.path.basename(classify_file),
                    "error": str(e)
                })

        #  todo: aggregate file result into "result" part of file_results

        per_file_results.append(file_results)

    os.makedirs(result_json_path, exist_ok=True)
    classify_aggregate_result = {}
    # todo: aggregate results here
    for result in per_file_results:
        continue

    processing_summary["status"] = "success"

    return processing_summary, direct_result, file_result
