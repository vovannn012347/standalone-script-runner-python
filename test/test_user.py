import json


def process(input, output):

	result =  {'status': 'test ok'}

	if output["out-file-result"]:
		with open(output["out-file-result"], "w", encoding='utf-8') as f:
			json.dump(result, f, indent=4)

	return result


