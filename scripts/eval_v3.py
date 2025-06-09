import asyncio
import base64
import json
import os
import re

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, BadRequestError, RateLimitError


load_dotenv()

api = AsyncAzureOpenAI(api_version="2025-04-01-preview")

OUTFILE = "data/eval_results_gpt41_1k.json"


async def eval_main():
    with open("data/tokyo-1k-compact.json") as f:
        answers = json.load(f)

    existing_results = {}

    try:
        with open(OUTFILE) as f:
            existing_results_list = json.load(f)
            existing_results = {entry["id"]: entry for entry in existing_results_list}
    except FileNotFoundError:
        pass

    answers_with_lamp = answers

    async def eval_one(
        i: int,
    ):
        """Evaluate a single entry."""
        answer = answers_with_lamp[i]

        ward = answer["city"]

        id = answer["id"]
        panoid = answer["panoId"]
        image_path = f"data/1k/{id}_{panoid}.jpg"

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
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
                          You will be given a scene image from Tokyo. Your task is to guess one of the 23 wards of Tokyo based on the image. Try to identify the image based on the key features and make a guess.

                          Your answer format should be an XML object with the following structure:
                          <observation>Details about the image without specifying the ward</observation><reasoning>Based on the observation, try to look for candidate wards that might match the image. If you are not sure, guess the ward that you think is most likely to match the image.</reasoning><ward>Ward name w/o suffix</ward>
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
        guessed_ward = match.group(3).strip()

        answer_ward = ward.lower().replace("ō", "o").replace("ē", "e")
        guessed_ward = (
            guessed_ward.lower()
            .replace("ō", "o")
            .replace("ē", "e")
            .replace(" ward", "")
            .replace("-ku", "")
            .replace("ku", "")
        )

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

        with open(OUTFILE, "w") as f:
            json.dump(sorted(existing_results.values(), key=lambda x: x["id"]), f, indent=2, ensure_ascii=False)

    tasks = []
    first = 0

    for i in range(len(answers_with_lamp)):
        if answers_with_lamp[i]["id"] in existing_results:
            continue

        if len(tasks) >= 20:
            try:
                await asyncio.gather(*tasks)
                tasks = []
            except RateLimitError as e:
                # message = e.message
                # match = re.search(r"retry after (\d+) seconds?", message)
                # retry_after = int(match.group(1)) if match else 60
                retry_after = 60
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
        tasks.append(eval_one(i))

    if len(tasks) > 0:
        await asyncio.gather(*tasks)
        with open(OUTFILE, "w") as f:
            json.dump(sorted(existing_results.values(), key=lambda x: x["id"]), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(eval_main())
    # asyncio.run(has_lamp_main())
