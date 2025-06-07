import base64
import json
import os
import re

from openai import OpenAI


api = OpenAI(api_key="0", base_url="http://0.0.0.0:8000/v1")


def extract_info(text: str):
    """Extract ward."""
    ward_match = re.search(r"<ward>\s*(.*?)\s*</ward>", text)
    return {
        "ward": ward_match.group(1) if ward_match else None,
    }


with open("data/tokyo_2k_v2_test.json") as f:
    dataset = json.load(f)

with open("../output_gpt.json") as f:
    answers = json.load(f)

DATAROOT = "data/"

FILE = "lora_3b_lamp"


def main():
    existing_results = []
    # Load existing evaluation results
    try:
        with open(f"../data/eval_results_{FILE}.json") as f:
            existing_results = json.load(f)
            evaluated_ids = {entry["id"] for entry in existing_results}

    except FileNotFoundError:
        evaluated_ids = set()

    entries = [entry for entry in dataset if int(entry["image"].split("/")[-1].split("_")[0]) not in evaluated_ids]

    # Evaluate each entry
    results = []
    for entry in entries:
        image_path = entry["image"].replace("tokyo_2k/", "tokyo_2k_v2/")

        id = int(image_path.split("/")[-1].split("_")[0])

        with open(os.path.join(DATAROOT, image_path), "rb") as img_file:
            image_data = img_file.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Call OpenAI API
        response = api.chat.completions.create(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Where is this place located in Tokyo? Provide observations of its key features, and based on that, reason about the ward it is located in. Your answer format should be an XML object with the following structure:  <observation>Details about the image without specifying the ward</observation><reasoning>Based on the observation, try to look for candidate wards that might match the image. If you are not sure, guess the ward that you think is most likely to match the image.</reasoning><ward>Ward name</ward>""",
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    ],
                }
            ],
        )

        entry_info = extract_info(entry["conversations"][1]["value"])
        expected_ward = entry_info.get("ward")

        result = {"id": id, "match": False, "expected": {"ward": expected_ward}}
        # Parse the response
        content = response.choices[0].message.content

        if not content:
            continue

        print(f"Evaluating ID: {id}")

        answer = extract_info(content)
        ward = answer.get("ward")

        # Check if the answer matches
        print(f"Evaluating ID: {id}, Ward: {ward} vs Expected Ward: {expected_ward}")

        match = ward == expected_ward

        result["predicted"] = {
            "ward": ward,
            "raw": content,
        }
        result["match"] = match
        results.append(result)

        # Append new results to existing ones
        if existing_results is not None:
            results.extend(existing_results)

        # Save the updated evaluation results
        with open(f"data/eval_results_{FILE}.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
