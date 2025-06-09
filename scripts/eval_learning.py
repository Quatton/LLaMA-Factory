import asyncio
import base64
import json
import os
import re
from enum import Enum

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, BadRequestError, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from openai.types.responses import EasyInputMessageParam, ResponseInputParam
from pydantic import BaseModel


load_dotenv()

api = AsyncAzureOpenAI(api_version="2025-04-01-preview")

OUTFILE = "data/eval_results_gpt41_jp_5k_ttt.json"
DATA = "data/japan-5k-compact.json"
IMAGE_ROOT = "data/jp"
OUTFILE_USEFUL_NOT_USEFUL = "data/useful_not_useful_jp_5k.json"


class Prefecture(str, Enum):
    AICHI = "aichi"
    AKITA = "akita"
    AOMORI = "aomori"
    CHIBA = "chiba"
    EHIME = "ehime"
    FUKUI = "fukui"
    FUKUOKA = "fukuoka"
    FUKUSHIMA = "fukushima"
    GIFU = "gifu"
    GUNMA = "gunma"
    HIROSHIMA = "hiroshima"
    HOKKAIDO = "hokkaido"
    HYOGO = "hyogo"
    IBARAKI = "ibaraki"
    ISHIKAWA = "ishikawa"
    IWATE = "iwate"
    KAGAWA = "kagawa"
    KAGOSHIMA = "kagoshima"
    KANAGAWA = "kanagawa"
    KOCHI = "kochi"
    KUMAMOTO = "kumamoto"
    KYOTO = "kyoto"
    MIE = "mie"
    MIYAGI = "miyagi"
    MIYAZAKI = "miyazaki"
    NAGANO = "nagano"
    NAGASAKI = "nagasaki"
    NARA = "nara"
    NIIGATA = "niigata"
    OITA = "oita"
    OKAYAMA = "okayama"
    OKINAWA = "okinawa"
    OSAKA = "osaka"
    SAGA = "saga"
    SAITAMA = "saitama"
    SHIGA = "shiga"
    SHIMANE = "shimane"
    SHIZUOKA = "shizuoka"
    TOCHIGI = "tochigi"
    TOKUSHIMA = "tokushima"
    TOKYO = "tokyo"
    TOTTORI = "tottori"
    TOYAMA = "toyama"
    WAKAYAMA = "wakayama"
    YAMAGATA = "yamagata"
    YAMAGUCHI = "yamaguchi"
    YAMANASHI = "yamanashi"


class UsefulNotUseful(BaseModel):
    usefuls: dict[str, str] | None = None
    not_usefuls: dict[str, str] | None = None

    class Config:
        use_enum_values = True
        extra = "forbid"


async def eval_main():
    with open(DATA) as f:
        answers = json.load(f)[:1000]  # Limit to 1000 entries for testing

    existing_results = {}
    usefuls = {}
    not_usefuls = {}

    try:
        with open(OUTFILE) as f:
            existing_results_list = json.load(f)
            existing_results = {entry["id"]: entry for entry in existing_results_list}
    except FileNotFoundError:
        pass

    try:
        with open(OUTFILE_USEFUL_NOT_USEFUL) as f:
            usefuls_not_usefuls = json.load(f)
            usefuls = usefuls_not_usefuls.get("usefuls", {})
            not_usefuls = usefuls_not_usefuls.get("not_usefuls", {})

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

        system: EasyInputMessageParam = {
            "role": "system",
            "content": f"""You are an expert in identifying Japanese prefectures based on images. These are the informations you have learned so far:

These are the useful cues you have learned so far. Meaning that these cues helped you guess prefectures correctly or at least mentioned about it in the reasoning despite not making it to the final guess:
{usefuls}

These are the not useful cues you have learned so far. Meaning that these cues do not help you guess prefectures correctly or at least distracted in the reasoning despite not making it to the final guess:
{not_usefuls}""",
        }

        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": [
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
            {"role": "assistant", "content": []},
        ]
        messages = messages[:-1]

        response = await api.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
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
        guessed_prefecture = match.group(3).strip()

        answer_prefecture = prefecture.lower().replace("ō", "o").replace("ē", "e")
        guessed_prefecture = (
            guessed_prefecture.lower()
            .replace("ō", "o")
            .replace("ē", "e")
            .replace(" prefecture", "")
            .replace("-ku", "")
            .replace("ku", "")
        )

        messages2: ResponseInputParam = [
            system,
            {
                "role": "assistant",
                "content": content,
            },
            {
                "role": "user",
                "content": f'The answer was {answer_prefecture}. If you guessed correctly, remember useful information for future guesses. If you guessed incorrectly, try to learn from your mistake and remember not useful information. Respond with the provided JSON format. This will be used to update your knowledge for future guesses.\n\n Usage guide: If you respond with {{{{ "usefuls": {{{{ "prefecture_a": "some useful information" }}}}, "not_usefuls": {{{{ "prefecture_b": "some not useful information" }}}} }}}}, it will **REPLACE** not just appending to the existing usefuls and not_usefuls, so you should be careful not to overwrite useful information that you have learned so far. Combine and reconcile. Lower case no suffix.\n\n',
            },
        ]
        resp = await api.responses.parse(
            model="gpt-4.1",
            input=messages2,
            text_format=UsefulNotUseful,
        )

        if resp.usage:
            print(f"Tokens used for {id}: {resp.usage.total_tokens}")

        response_json = resp.output_parsed

        if response_json:
            # update usefuls and not_usefuls
            if response_json.usefuls:
                for prefecture, useful in response_json.usefuls.items():
                    usefuls[prefecture] = useful.strip()
                    print(f"I learned useful information for {prefecture}: {useful.strip()}")
            if response_json.not_usefuls:
                for prefecture, not_useful in response_json.not_usefuls.items():
                    not_usefuls[prefecture] = not_useful.strip()
                    print(f"I learned not useful information for {prefecture}: {not_useful.strip()}")

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

        with open(OUTFILE_USEFUL_NOT_USEFUL, "w") as f:
            json.dump(
                {
                    "usefuls": {k: v for k, v in usefuls.items() if v.strip()},
                    "not_usefuls": {k: v for k, v in not_usefuls.items() if v.strip()},
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    tasks = []
    first = 0

    for i in range(len(answers_with_lamp)):
        if answers_with_lamp[i]["id"] in existing_results:
            continue

        if len(tasks) >= 1:
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
