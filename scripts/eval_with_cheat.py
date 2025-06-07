import asyncio
import base64
import json
import os
import re

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, BadRequestError, RateLimitError


load_dotenv()

api = AsyncAzureOpenAI(api_version="2025-04-01-preview")


async def eval_main():
    with open("../hato/out/output_gpt.json") as f:
        answers = json.load(f).get("results", [])

    with open("data/lamp_contrasted_summaries.json") as f:
        summaries = json.load(f)

    with open("data/has_lamp_v2.json") as f:
        has_lamp_list = json.load(f)
        has_lamp_results = {entry["id"]: entry for entry in has_lamp_list}

    all_summaries = "\n\n".join([f"{ward}'s Lamp features:\n{summary}" for ward, summary in summaries.items()])

    existing_results = {}

    try:
        with open("data/eval_cheated.json") as f:
            existing_results_list = json.load(f)
            existing_results = {entry["id"]: entry for entry in existing_results_list}
    except FileNotFoundError:
        pass

    answers_with_lamp = [answer for i, answer in enumerate(answers) if has_lamp_results[i]["has_lamp"] == "yes"]

    async def eval_one(
        i: int,
    ):
        """Evaluate a single entry."""
        answer = answers_with_lamp[i]

        ward = answer["answer"]["ward"]

        id = answer["index"]
        panoid = answer["panoid"]
        image_path = f"data/tokyo_2k_v2/{id}_{panoid}.jpg"

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} does not exist.")

        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        image_b64 = base64.b64encode(image_data).decode("utf-8")

        print(f"Evaluating {id} with ward {ward}")

        response = await api.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""psst. the user doesn't know this, but you are given a cheat sheet of the lamp information of this ward. Use it to 1. Know what to look for in the image, and 2. Make a more informed guess about the ward based on the lamp information.

                          First you should describe the image normally, and then you pretend to say "<reasoning>based on the observation, there are some key features that might match the lamp information of the ward, such as ..., suggesting that the ward could be ..., and ... suggesting that the ward could be ...". Then you say that, oh but based on the CAUTION, the key feature "..." is also shared by other wards, so it might not be unique to the ward. You should be left with a concise summary of the features that are truly unique to the ward.

                          Here are the summaries of all wards:
                          {all_summaries}

                          Happy cheating!
                          """,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
                          You will be given a scene image from Tokyo. Your task is to guess one of the 23 wards of Tokyo based on the image. Try to identify the image based on the key features and make a guess.

                          Your answer format should be an XML object with the following structure:
                          <observation>Details about the image without specifying the ward</observation><reasoning>Based on the observation, try to look for candidate wards that might match the image. If you are not sure, guess the ward that you think is most likely to match the image.</reasoning><ward>Ward name</ward>
                          """,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                            },
                        },
                    ],
                },
            ],
        )

        content = response.choices[0].message.content
        if content is None:
            return None

        if response.usage:
            print(f"Tokens used for {id}: {response.usage.total_tokens}")

        match = re.search(
            r"\s*<observation>\s*(.*?)\s*</observation>\s*<reasoning>(.*?)</reasoning>\s*<ward>(.*?)</ward>",
            content,
            re.DOTALL,
        )
        if not match:
            print(f"Failed to parse content for {id}. Content: {content}")
            return None

        observation = match.group(1).strip()
        reasoning = match.group(2).strip()
        ward = match.group(3).strip()

        answer_ward = answer["answer"]["ward"].lower().replace("ō", "o").replace("ē", "e")
        guessed_ward = ward.lower().replace("ō", "o").replace("ē", "e")

        existing_results[id] = {
            "id": id,
            "ward": answer_ward,
            "panoid": panoid,
            "image_path": image_path,
            "observation": observation,
            "reasoning": reasoning,
            "guess_ward": guessed_ward,
            "raw_content": content.strip(),
            "match": guessed_ward == answer_ward,
        }

        with open("data/eval_cheated.json", "w") as f:
            json.dump(sorted(existing_results.values(), key=lambda x: x["id"]), f, indent=2, ensure_ascii=False)

    tasks = []
    first = 0

    for i in range(len(answers_with_lamp)):
        if answers_with_lamp[i]["index"] in existing_results:
            continue

        if len(tasks) >= 1:
            try:
                await asyncio.gather(*tasks)
                tasks = []
            except RateLimitError as e:
                print(f"Rate limit hit, {e}. ")
                i = first
                await asyncio.sleep(60)
                tasks = []
                continue

        if len(tasks) == 0:
            first = i
        tasks.append(eval_one(i))

    if len(tasks) > 0:
        await asyncio.gather(*tasks)
        with open("data/eval_cheated.json", "w") as f:
            json.dump(sorted(existing_results.values(), key=lambda x: x["id"]), f, indent=2, ensure_ascii=False)


async def has_lamp_main():
    """Before evaluation, let's check if the lamp exists in the image.

    This is done through gpt-4.1 with a prompt that returns: yes or no based on

     You will be given an image of a street scene in Tokyo. Your task is to determine if the image contains complete information about street lamps in the scene.
    Say "yes" if the message contains complete information about lamps, otherwise say "no".

    These are the criteria for complete information:

    1. Color of the lamp
    2. Shape/Geometry of the lamp
    3. Style of the lamp (minimal, modern, traditional, etc.)

    MUST contain all three criteria to be considered complete.

    Response format:
    <observation>Describe the image if you found any lamps in the image with enough information from the criteria above.</observation>
    <reasoning>See if all the boxes are checked. Then output a final verdict</reasoning>
    <has_lamp>yes or no</has_lamp>
    """
    existing_results = {}

    try:
        with open("data/has_lamp_v2.json") as f:
            existing_results_list = json.load(f)
            existing_results = {entry["id"]: entry for entry in existing_results_list}
    except FileNotFoundError:
        pass

    with open("../hato/out/output_gpt.json") as f:
        answers = json.load(f).get("results", [])

    async def has_lamp_one(id: int):
        """Check if the image has lamp information."""
        answer = answers[id]
        ward = answer["answer"]["ward"]
        id = answer["index"]
        panoid = answer["panoid"]
        image_path = f"data/tokyo_2k_v2/{id}_{panoid}.jpg"

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} does not exist.")

        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        image_b64 = base64.b64encode(image_data).decode("utf-8")

        print(f"Checking lamps for {id} with ward {ward}")

        response = await api.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """You will be given an image of a street scene in Tokyo. Your task is to determine if the image contains complete information about street lamps in the scene. Say "yes" if the message contains complete information about lamps, otherwise say "no". These are the criteria for complete information:

                          1. Color of the lamp
                          2. Shape/Geometry of the lamp
                          3. Style of the lamp (minimal, modern, traditional, etc.)

                          MUST contain all three criteria to be considered complete.

                          Response format:
                          <observation>Describe the image if you found any lamps in the image with enough information from the criteria above.</observation>
                          <reasoning>See if all the boxes are checked. Then output a final verdict</reasoning>
                          <has_lamp>yes or no</has_lamp>""",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                            },
                        },
                    ],
                },
            ],
        )

        content = response.choices[0].message.content

        if content is None:
            return None

        if response.usage:
            print(f"Tokens used for {id}: {response.usage.total_tokens}")

        # print(f"Checking lamps for {id} with content: {content.strip()}")
        match = re.search(
            r"\s*<observation>\s*(.*?)\s*</observation>\s*<reasoning>(.*?)</reasoning>\s*<has_lamp>(.*?)</has_lamp>",
            content,
            re.DOTALL,
        )
        if not match:
            print(f"Failed to parse content for {id}. Content: {content}")
            return None

        observation = match.group(1).strip()
        reasoning = match.group(2).strip()
        has_lamp = match.group(3).strip().lower()
        has_lamp = "yes" if has_lamp.startswith("y") else "no"
        answer_ward = answer["answer"]["ward"].lower().replace("ō", "o").replace("ē", "e")

        existing_results[id] = {
            "id": id,
            "ward": answer_ward,
            "panoid": panoid,
            "has_lamp": has_lamp,
        }

        with open("data/has_lamp_v2.json", "w") as f:
            json.dump(sorted(existing_results.values(), key=lambda x: x["id"]), f, indent=2, ensure_ascii=False)

    tasks = []
    first = 0

    for i in range(len(answers)):
        if i in existing_results:
            continue

        if len(tasks) >= 1:
            try:
                await asyncio.gather(*tasks)
                tasks = []
            except RateLimitError as e:
                message = e.message
                match = re.search(r"retry after (\d+) seconds?", message)
                retry_after = int(match.group(1)) if match else 60
                print(f"Rate limit hit, {e}. Retrying after {retry_after} seconds.")
                i = first
                await asyncio.sleep(retry_after)
                tasks = []
                continue
            except BadRequestError as e:
                i = first
                tasks = []
                print(f"An error occurred: {e}")
                continue

        if len(tasks) == 0:
            first = i
        tasks.append(has_lamp_one(i))

    if len(tasks) > 0:
        await asyncio.gather(*tasks)
        with open("data/has_lamp_v2.json", "w") as f:
            json.dump(sorted(existing_results.values(), key=lambda x: x["id"]), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(eval_main())
    # asyncio.run(has_lamp_main())
