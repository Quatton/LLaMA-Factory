import base64
import json
import os
import re

from openai import OpenAI


api = OpenAI(api_key="0", base_url="http://0.0.0.0:8000/v1")


def extract_info(text: str):
    """Extract ward and town from the text."""
    ward_match = re.search(r"<ward>\s*(.*?)\s*</ward>", text)
    town_match = re.search(r"<town>\s*(.*?)\s*</town>", text)

    ward = ward_match.group(1).strip() if ward_match else "Unknown"
    town = town_match.group(1).strip() if town_match else None

    return {"ward": ward, "town": town}


with open("data/tokyo_2k_test.json") as f:
    dataset = json.load(f)

with open("../output_gpt.json") as f:
    answers = json.load(f)

DATAROOT = "data/"


def main():
    existing_results = []
    # Load existing evaluation results
    try:
        with open("../data/eval_results.json") as f:
            existing_results = json.load(f)
            evaluated_ids = {entry["id"] for entry in existing_results}

    except FileNotFoundError:
        evaluated_ids = set()

    entries = [entry for entry in dataset if int(entry["image"].split("/")[-1].split("_")[0]) not in evaluated_ids]

    # Evaluate each entry
    results = []
    for entry in entries:
        image_path = entry["image"]

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
                            "text": """Where is this location in Tokyo? Provide observations and reasoning.

Response format:

<observation>
Things to describe the place, such as buildings, landmarks, etc. Do not include the name of ward or town.
</observation>
<reasoning>
Reasoning about the location, based on the observation.
</reasoning>
<ward>Choose one from 23 wards</ward>

23 wards in Tokyo: <ward>Adachi</ward>, <ward>Arakawa</ward>, <ward>Bunkyo</ward>, <ward>Chiyoda</ward>, <ward>Chuo</ward>, <ward>Edogawa</ward>, <ward>Itabashi</ward>, <ward>Katsushika</ward>, <ward>Kita</ward>, <ward>Koto</ward>, <ward>Meguro</ward>, <ward>Minato</ward>, <ward>Nerima</ward>, <ward>Ota</ward>, <ward>Setagaya</ward>, <ward>Shibuya</ward>, <ward>Shinagawa</ward>, <ward>Shinjuku</ward>, <ward>Suginami</ward>, <ward>Toshima</ward>, <ward>Sumida</ward>, <ward>Taito</ward>

Common mistakes:
<ward>Tokyo</ward> is not a valid ward. Tokyo is a prefecture.""",
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    ],
                }
            ],
        )

        entry_info = extract_info(entry["conversations"][1]["value"])
        expected_ward = entry_info.get("ward")
        expected_town = entry_info.get("town")

        result = {"id": id, "match": False, "expected": {"ward": expected_ward, "town": expected_town}}
        # Parse the response
        content = response.choices[0].message.content

        if not content:
            results.append(result)
            continue

        print(f"Evaluating ID: {id}, Content: {content}")

        answer = extract_info(content)
        ward = answer.get("ward")
        town = answer.get("town")

        # Check if the answer matches
        print(
            f"Evaluating ID: {id}, Ward: {ward}, Town: {town} vs Expected Ward: {expected_ward}, Expected Town: {expected_town}"
        )

        match = ward == expected_ward

        result["predicted"] = {
            "ward": ward,
            "town": town,
            "observation": answer.get("observation", ""),
            "reasoning": answer.get("reasoning", ""),
            "raw": content,
        }
        result["match"] = match
        results.append(result)

        # Append new results to existing ones
        if existing_results is not None:
            results = existing_results + results

        # Save the updated evaluation results
        with open("data/eval_results_base_3b.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
