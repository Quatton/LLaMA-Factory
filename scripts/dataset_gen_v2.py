import json


with open("data/lamp_reguessing_results_small.json") as f:
    data = json.load(f)
    results = data

with open("data/tokyo_lamps.json", "w") as f:
    entries = []
    for result in results:
        user_msg = {
            "from": "human",
            "value": "Where is this place located in Tokyo? Provide observations of its key features, and based on that, reason about the ward it is located in. Your answer format should be an XML object with the following structure:  <observation>Details about the image without specifying the ward</observation><reasoning>Based on the observation, try to look for candidate wards that might match the image. If you are not sure, guess the ward that you think is most likely to match the image.</reasoning><ward>Ward name</ward> <image>",
        }
        assistant_msg = {"from": "gpt", "value": result["raw_content"]}
        conversations = [user_msg, assistant_msg]
        image = f"tokyo_2k/{result['id']}_{result['panoid']}.jpg"
        data = {
            "conversations": conversations,
            "image": image,
        }
        entries.append(data)

    json.dump(entries, f, indent=2, ensure_ascii=False)
