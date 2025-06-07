import json
import re


with open("../output_gpt.json") as f:
    data = json.load(f)
    results = data["results"]

with open("data/tokyo_2k.json", "w") as f:
    entries = []
    for result in results[:-400]:
        user_msg = {
            "from": "human",
            "value": "Where is this place located in Tokyo? Provide observations and reasoning. <image>",
        }
        assistant_msg = {
            "from": "gpt",
            "value": re.sub(
                r"<answer>\s*<ward>\s*(.*?)\s*</ward>\s*<town>\s*.*\s*</town>\s*</answer>",
                r"<ward>\1</ward>",
                result["answer"]["raw"],
            ),
        }
        conversations = [user_msg, assistant_msg]
        image = f"tokyo_2k/{result['index']}_{result['panoid']}.jpg"
        data = {
            "conversations": conversations,
            "image": image,
        }
        entries.append(data)

    json.dump(entries, f, indent=2, ensure_ascii=False)

with open("../output.json") as f:
    bad_data = json.load(f)
    bad_results = bad_data["results"]

with open("data/tokyo_2k_dpo.json", "w") as f:
    dpo_entries = []
    for good_result, bad_result in zip(entries, bad_results):
        conversations = good_result["conversations"][0:1]
        chosen = good_result["conversations"][1]
        rejected = {
            "from": "gpt",
            "value": f"""<observation>{bad_result["answer"]["observation"]}</observation>
<reasoning>{bad_result["answer"]["reasoning"]}</reasoning>
<answer><ward>{bad_result["answer"]["ward"]}</ward></answer>""",
        }
        data = {
            "conversations": conversations,
            "chosen": chosen,
            "rejected": rejected,
            "image": good_result["image"],
        }
        dpo_entries.append(data)

    json.dump(dpo_entries, f, indent=2, ensure_ascii=False)
