# The executor handles the dynamic import of 'logic' or other labels
# import logic  # This works because the executor added it to sys.path

def process(input, output):
    # 'input' contains paths and arrays
    # 'output' contains resolved folder and file paths
    
    # Example logic using the imported module
    result_data = logic.analyze(input['input-image'])
    
    # Saving to a specific output_file
    with open(output['out-file-result'], 'w') as f:
        json.dump(result_data, f)
        
    return {
        "status": "success",
        "artifacts": [
            { "label": "out-file-result", "file": os.path.basename(output['out-file-result']) }
        ]
    }