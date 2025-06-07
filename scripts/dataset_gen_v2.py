import json


with open("data/lamp_reguessing_results_small.json") as f:
    data = json.load(f)
    results = data

# with open("data/tokyo_lamps.json", "w") as f:
#     entries = []
#     for result in results:
#         user_msg = {
#             "from": "human",
#             "value": "Where is this place located in Tokyo? Provide observations of its key features, and based on that, reason about the ward it is located in. Your answer format should be an XML object with the following structure:  <observation>Details about the image without specifying the ward</observation><reasoning>Based on the observation, try to look for candidate wards that might match the image. If you are not sure, guess the ward that you think is most likely to match the image.</reasoning><ward>Ward name</ward> <image>",
#         }
#         assistant_msg = {"from": "gpt", "value": result["raw_content"]}
#         conversations = [user_msg, assistant_msg]
#         image = f"tokyo_2k/{result['id']}_{result['panoid']}.jpg"
#         data = {
#             "conversations": conversations,
#             "image": image,
#         }
#         entries.append(data)

#     json.dump(entries, f, indent=2, ensure_ascii=False)

with open("../output_gpt.json") as f:
    output = json.load(f)["results"]

with open("data/has_lamp_v2.json") as f:
    has_lamp_data = json.load(f)

with open("data/tokyo_2k_v2_test.json", "w") as f:
    entries = []
    output_with_lamp = [result for result in output if has_lamp_data[result["index"]]["has_lamp"] == "yes"]

    for result in output_with_lamp:
        entries.append(
            {
                "id": result["index"],
                "panoid": result["panoid"],
                "image": f"tokyo_2k_v2/{result['index']}_{result['panoid']}.jpg",
                "ward": result["answer"]["ward"].strip().lower().replace("ō", "o").replace("ē", "e"),
            }
        )
    json.dump(entries, f, indent=2, ensure_ascii=False)
