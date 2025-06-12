import asyncio
import base64
import json
import os
import re

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, BadRequestError, RateLimitError


load_dotenv()

api = AsyncAzureOpenAI(api_version="2025-04-01-preview")

OUTFILE = "data/eval_results_gpt41_jp_sampled_p_02_top_3.json"
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

    answers_with_lamp = answers

    async def eval_one(
        i: int,
    ):
        """Evaluate a single entry."""
        answer = answers_with_lamp[i]

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
            messages=[
                #                 {
                #                     "role": "system",
                #                     "content": """
                #                     You are an expert in identifying Japanese prefectures based on images. Your task is to analyze the provided image and guess the prefecture it represents. Pay attention to key features, landmarks, and cultural elements that might indicate the prefecture. Try to include as many candidate as possible at the beginning, then finally narrow it down by intersection
                #                     Topography & Landscape
                # Predominantly Mountainous / Hilly Terrain
                # Prefectures: Aomori, Iwate, Akita, Yamagata, Fukushima, Gunma, Nagano, Yamanashi, Gifu, Mie, Nara, Wakayama, Tottori, Shimane, Hiroshima, Yamaguchi, Tokushima, Ehime, Kochi, Fukuoka, Nagasaki, Kumamoto, Oita, Miyazaki, Kagoshima
                # Expansive Flat Plains / Lowlands
                # Prefectures: Hokkaido, Miyagi, Ibaraki, Tochigi, Saitama, Chiba, Niigata, Toyama, Aichi, Saga
                # Mixed Terrain (Significant Flatlands and Mountains/Hills)
                # Prefectures: Miyagi, Gunma, Tochigi, Saitama, Chiba, Kanagawa, Niigata, Toyama, Ishikawa, Shizuoka, Aichi, Shiga, Kyoto, Osaka, Hyogo, Okayama, Fukuoka, Kumamoto
                # Coastal Features / Proximity to Sea
                # Prefectures: Hokkaido, Miyagi, Chiba, Kanagawa, Toyama, Shizuoka, Aichi, Mie, Wakayama, Hyogo, Shimane, Hiroshima, Yamaguchi, Kagawa, Ehime, Fukuoka, Nagasaki, Kumamoto, Oita, Kagoshima
                # Urbanization & Population Density
                # Predominantly Rural / Sparsely Populated
                # Prefectures: Hokkaido, Aomori, Iwate, Akita, Yamagata, Fukushima, Niigata, Yamanashi, Nagano, Tottori, Shimane, Tokushima, Kochi, Saga
                # Primarily Suburban / Semi-Rural
                # Prefectures: Miyagi, Ibaraki, Tochigi, Gunma, Kanagawa, Ishikawa, Fukui, Gifu, Shiga, Kyoto, Nara, Wakayama, Okayama, Nagasaki, Kumamoto, Oita, Miyazaki, Kagoshima
                # Highly Urbanized / Densely Populated Areas
                # Prefectures: Saitama, Chiba, Tokyo, Kanagawa, Aichi, Osaka, Hyogo, Fukuoka
                # Presence of Industrial Zones / Factories
                # Prefectures: Gunma, Saitama, Chiba, Kanagawa, Ishikawa, Shizuoka, Aichi, Gifu, Hyogo, Hiroshima, Fukuoka
                # Agriculture
                # Extensive Rice Paddies
                # Prefectures: Hokkaido, Aomori, Iwate, Akita, Fukushima, Ibaraki, Tochigi, Gunma, Chiba, Niigata, Toyama, Gifu, Aichi, Mie, Shiga, Kyoto, Hyogo, Nara, Wakayama, Tottori, Okayama, Hiroshima, Yamaguchi, Tokushima, Kagawa, Ehime, Saga, Kumamoto, Oita, Miyazaki, Kagoshima
                # Presence of Greenhouses
                # Prefectures: Yamagata, Ibaraki, Tochigi, Gunma, Chiba, Niigata, Toyama, Nagano, Gifu, Tokushima, Kochi, Saga, Miyazaki, Kagoshima
                # Orchards / Fruit Cultivation (including citrus)
                # Prefectures: Aomori, Fukushima, Hiroshima, Fukuoka
                # Climate & Environment
                # Adapted for Cold / Snowy Winters
                # Prefectures: Hokkaido, Aomori, Iwate, Akita, Yamagata, Gunma, Niigata, Toyama, Ishikawa, Fukui, Nagano, Okayama
                # Warm / Humid / Subtropical Elements
                # Prefectures: Gifu, Shizuoka, Wakayama, Tottori, Kagawa, Nagasaki, Miyazaki, Kagoshima
                # Dense Forests / Lush Greenery
                # Prefectures: Aomori, Iwate, Akita, Yamagata, Fukushima, Tochigi, Yamanashi, Nagano, Gifu, Mie, Nara, Wakayama, Tottori, Shimane, Hiroshima, Yamaguchi, Tokushima, Ehime, Kochi, Nagasaki, Kumamoto, Oita, Miyazaki, Kagoshima
                # Architecture & Infrastructure
                # Primarily Traditional Japanese Houses (Tiled/Sloped Roofs)
                # Prefectures: Fukui, Ishikawa, Kyoto, Niigata, Tottori, Shimane, Mie, Nara, Wakayama
                # Mix of Modern and Traditional Housing
                # Prefectures: Aomori, Iwate, Miyagi, Fukushima, Ibaraki, Tochigi, Saitama, Chiba, Tokyo, Kanagawa, Toyama, Shiga, Osaka, Hyogo, Okayama, Hiroshima, Yamaguchi, Kagawa, Ehime, Fukuoka, Nagasaki, Kumamoto, Oita, Miyazaki, Kagoshima
                # Ubiquitous Overhead Utility Poles / Wires
                # (This is a common feature across almost all listed prefectures) Prefectures: Aomori, Iwate, Akita, Miyagi, Ibaraki, Saitama, Chiba, Tokyo, Kanagawa, Niigata, Toyama, Ishikawa, Fukui, Yamanashi, Nagano, Shizuoka, Aichi, Mie, Shiga, Kyoto, Osaka, Hyogo, Nara, Wakayama, Tottori, Okayama, Yamaguchi, Tokushima, Kagawa, Ehime, Nagasaki, Kumamoto, Oita, Miyazaki, Kagoshima
                # Narrow Streets / Roads
                # Prefectures: Aomori, Iwate, Fukushima, Ibaraki, Saitama, Tokyo, Kanagawa, Aichi, Mie, Shiga, Osaka
                #                     """,
                #                 },
                # {
                #     "role": "system",
                #     "content": "There is no Tokyo here so don't answer Tokyo.",
                # },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
                          You will be given a scene image from Japan. Your task is to guess one of the 47 prefectures of Tokyo based on the image. Try to identify the image based on the key features and make a guess. You can guess at least 3 candidates, comma-separated, no spaces.

                          Your answer format should be an XML object with the following structure:
                          <observation>Details about the image without specifying the prefecture</observation><reasoning>Based on the observation, try to look for candidate prefectures that might match the image. If you are not sure, guess the prefecture that you think is most likely to match the image.</reasoning><prefecture>prefecture1,prefecture2,prefecture3</prefecture>
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
            top_p=0.2,
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
        if not match:
            print(f"Failed to parse content for {id}. Content: {content}")
            return None

        observation = match.group(1).strip()
        reasoning = match.group(2).strip()
        guessed_prefectures = match.group(3).strip().split(",")

        answer_prefecture = prefecture.lower().replace("ō", "o").replace("ē", "e")
        guessed_prefectures = [
            guessed_prefecture.lower().strip().replace("ō", "o").replace("ē", "e").replace(" prefecture", "")
            for guessed_prefecture in guessed_prefectures
        ]

        existing_results[id] = {
            "id": id,
            "prefecture": answer_prefecture,
            "panoid": panoid,
            "image_path": image_path,
            "observation": observation,
            "reasoning": reasoning,
            "guess_prefecture": guessed_prefectures,
            "raw_content": content.strip(),
            "match": answer_prefecture in guessed_prefectures,
        }

        with open(OUTFILE, "w") as f:
            json.dump(sorted(existing_results.values(), key=lambda x: x["id"]), f, indent=2, ensure_ascii=False)

    tasks = []
    first = 0

    for i in range(len(answers_with_lamp)):
        if answers_with_lamp[i]["id"] in existing_results:
            continue

        if len(tasks) >= 40:
            try:
                await asyncio.gather(*tasks)
                tasks = []
            except RateLimitError as e:
                # message = e.message
                # match = re.search(r"retry after (\d+) seconds?", message)
                # retry_after = int(match.group(1)) if match else 60
                retry_after = 60
                print(f"Rate limit hit, {e}. Retrying after {retry_after} seconds.")
                i = 0
                await asyncio.sleep(retry_after)
                tasks = []
                continue
            except BadRequestError as e:
                i = 0
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
