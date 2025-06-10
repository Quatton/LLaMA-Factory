import asyncio
import base64
import json
import os
import re
from openai import AsyncOpenAI, BadRequestError, RateLimitError

api = AsyncOpenAI(api_key="0", base_url="http://0.0.0.0:8000/v1")

OUTFILE = os.environ.get("OUTFILE", "data/eval_results_lora_jp_sampled.json")
DATA = "data/japan-5k-sampled.json"
IMAGE_ROOT = "data/jp-test"

async def eval_main():
    with open(DATA) as f:
        answers = json.load(f)

    existing_results = {}

    try:
        with open(OUTFILE) as f:
            existing_results_list = json.load(f)
            existing_results = {entry["id"]: entry for entry in existing_results_list}
    except FileNotFoundError:
        pass

    answer_wo_tokyo = [
        answer for answer in answers if answer["prefecture"].lower() != "tokyo"
    ]

    async def eval_one(
        i: int,
    ):
        """Evaluate a single entry."""
        answer = answer_wo_tokyo[i]

        prefecture = answer["prefecture"]

        id = answer["id"]
        panoid = answer["panoId"]
        image_path = f"{IMAGE_ROOT}/{id}_{panoid}.jpg"

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} does not exist.")

        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        image_b64 = base64.b64encode(image_data).decode("utf-8")

        print(f"Evaluating {id} with prefecture {prefecture}")

        response = await api.chat.completions.create(
            model="gpt-4.1",
            top_p=0.1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hint: There is no Tokyo in the samples so don't guess Tokyo."  
                        },
                        {
                            "type": "text",
                            "text": """
                          You will be given a scene image from Japan. Your task is to guess one of the 47 prefectures of Tokyo based on the image. Try to identify the image based on the key features and make a guess.

                          Your answer format should be an XML object with the following structure:
                          <observation>Details about the image without specifying the prefecture</observation><reasoning>Based on the observation, try to look for candidate prefectures that might match the image. If you are not sure, guess the prefecture that you think is most likely to match the image.</reasoning><prefecture>prefecture name w/o suffix or special characters</prefecture>
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
            r"\s*<observation>\s*(.*?)\s*</observation>\s*<reasoning>(.*?)</reasoning>\s*<prefecture>(.*?)</prefecture>",
            content,
            re.DOTALL,
        )
        
        observation = ""
        reasoning = ""
        guessed_prefecture = ""

        if match:
            observation = match.group(1).strip()
            reasoning = match.group(2).strip()
            guessed_prefecture = match.group(3).strip()
        else:
            match_prefecture = re.search(
                r"\s*<prefecture>(.*?)</prefecture>",
                content,
                re.DOTALL,
            )
            guessed_prefecture = match_prefecture.group(1).strip() if match_prefecture else ""
            
        answer_prefecture = prefecture.lower().replace("ō", "o").replace("ē", "e")
        guessed_prefecture = guessed_prefecture.lower().replace("ō", "o").replace("ē", "e").replace(" prefecture", "")

        existing_results[id] = {
            "id": id,
            "prefecture": answer_prefecture,
            "panoid": panoid,
            "image_path": image_path,
            "observation": observation,
            "reasoning": reasoning,
            "guess_prefecture": guessed_prefecture,
            "raw_content": content.strip(),
            "match": guessed_prefecture == answer_prefecture,
        }

        with open(OUTFILE, "w") as f:
            json.dump(sorted(existing_results.values(), key=lambda x: x["id"]), f, indent=2, ensure_ascii=False)

    tasks = []
    first = 0

    for i in range(len(answer_wo_tokyo)):
        if answer_wo_tokyo[i]["id"] in existing_results:
            continue

        if len(tasks) >= 5:
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
