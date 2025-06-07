import base64
import json
import os
import re
from typing import Literal

from openai import OpenAI


def extract_info(text: str):
    """Extract ward and town from the text."""
    ward_match = re.search(r"<ward>\s*(.*?)\s*</ward>", text)
    town_match = re.search(r"<town>\s*(.*?)\s*</town>", text)

    ward = ward_match.group(1).strip() if ward_match else "Unknown"
    town = town_match.group(1).strip() if town_match else None

    return {"ward": ward, "town": town}


api = OpenAI(api_key="0", base_url="http://0.0.0.0:8000/v1")


with open("../output_gpt.json") as f:
    answers = json.load(f).get("results", [])


def main():
    results_dict = {}

    try:
        with open("data/has_lamp_results.json") as f:
            existing_results = json.load(f)
            results_dict = {entry["id"]: entry for entry in existing_results}
    except FileNotFoundError:
        pass

    entries = answers.copy()

    for i, entry in enumerate(entries):
        id = entry["index"]

        result = results_dict.get(id, {"id": id, "has_lamp": None})

        if "ward" in result and "has_lamp" in result and result["has_lamp"] is not None:
            # print(f"Skipping ID {id} as it has already been processed.")
            continue

        if result["has_lamp"] is None:
            response = api.chat.completions.create(
                model="Qwen/Qwen3-4B",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""
                              You will be given an image description of a street scene in Tokyo. Your task is to determine if the description contains complete information about street lamps in the scene.
                              Say "yes" if the message contains complete information about lamps, otherwise say "no".

                              These are the criteria for complete information:

                              1. Color of the lamp
                              2. Shape/Geometry of the lamp
                              3. Style of the lamp (minimal, modern, traditional, etc.)

                              MUST contain all three criteria to be considered complete.

                              Image description:
                              {entry["answer"]["observation"]}""",
                            },
                        ],
                    }
                ],
            )

            content = response.choices[0].message.content

            print(f"Processing ID {id}: {content}")

            if not content:
                print(f"No content for ID {id}")
                continue

            answer = re.sub(r"<think>[\s\S]*?</think>", "", content).strip().lower()

            if answer not in ["yes", "no"]:
                print("Unexpected answer")
                continue

            result["has_lamp"] = answer == "yes"

        if "ward" not in result:
            result["ward"] = entry["answer"].get("ward", "Unknown")
        else:
            continue

        results_dict[id] = result

        results = list(results_dict.values())
        results.sort(key=lambda x: x["id"])

        with open("data/has_lamp_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
