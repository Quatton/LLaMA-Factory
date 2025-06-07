import asyncio
import base64
import itertools
import json
import os
import re

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, RateLimitError


load_dotenv()

api = AsyncAzureOpenAI(api_version="2025-04-01-preview")


with open("data/has_lamp_results.json") as f:
    lamps = json.load(f)

with open("../hato/out/output_gpt.json") as f:
    answers = json.load(f).get("results", [])

wards = {}


def schedule():
    count = 0
    total = len(lamps)
    for entry in lamps:
        if entry["has_lamp"]:
            ward_lower = entry["ward"].lower()
            if ward_lower not in wards:
                wards[ward_lower] = []

            wards[ward_lower].append(entry)

    print(f"{'Ward':<20} {'Entries':<10}")
    for ward, entries in wards.items():
        count += len(entries)
        print(f"{ward:<20} {len(entries):<10}")

    print(f"Total entries with lamps: {count} / {total}")

    # count = 0

    # for combo in itertools.combinations(wards, 2):
    #     ward1, ward2 = combo
    #     entries1 = wards[ward1]
    #     entries2 = wards[ward2]

    #     for entry1, entry2 in itertools.product(entries1, entries2):
    #         count += 1

    # print("Total combinations of entries from different wards:", count)


def combine_observations(relabeled_lamps):
    combined_results = []

    precombied_lamps = {ward: [] for ward in wards.keys()}
    for lamp in relabeled_lamps:
        ward = lamp["ward"].lower()
        precombied_lamps[ward].append(lamp["relabeling"])

    for ward, entries in wards.items():
        combined_observation = "\n\n".join(precombied_lamps[ward.lower()])
        combined_results.append(
            {
                "ward": ward,
                "observation": combined_observation,
            }
        )

    with open("data/combined_observations.json", "w") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)

    return combined_results


async def get_lamp_info(ward: str, observation: str):
    print(f"Processing ward: {ward} with observation length: {len(observation)}")
    response = await api.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                        You will be given a large text containing multiple observations about street lamps in a ward of Tokyo called: {ward}. Your task is to cross referencing each observation and find a common pattern or information about the street lamps in that ward. Besure to collect every distinct information about the street lamps in the ward, such as color, shape, style, and any other relevant details. And, after exiting your thought process, make a summary of the information you found. If there are more than one distinct pattern of lamps, you should make them into multiple groups and MUST not miss any key feature (for example you might find green lamp in Setagaya with oblong shape, but also lime-tinted lamp in another shape as well so you need to include both). Hint: Most of the lamp details are usually mentioned in the section of Municipal Utilities

                        Observation: {observation}
                        """,
                    }
                ],
            }
        ],
    )

    content = response.choices[0].message.content

    if content is None:
        raise ValueError("No content returned from the API.")

    return content


async def schedule_api_calls(combined_results):
    existing_wards = set()
    results_dict = {}

    try:
        with open("data/lamp_info.json") as f:
            existing_results = json.load(f)
            results_dict = {entry["ward"].lower(): entry["lamp_info"] for entry in existing_results}
            existing_wards = results_dict.keys()
    except FileNotFoundError:
        pass

    # if existing_wards has all entries for ward in combined_results, skip the entire processing
    if all(entry["ward"].lower() in existing_wards for entry in combined_results):
        print("All wards already processed. Skipping API calls.")
        return results_dict

    print(f"Existing wards: {existing_wards}")

    for entry in combined_results:
        if entry["ward"].lower() in existing_wards:
            print(f"Skipping {entry['ward']} as it is already processed.")
            continue
        ward = entry["ward"]
        observation = entry["observation"]
        obv = await get_lamp_info(ward, observation)
        print(f"Processed {ward} with observation: {obv}")
        results_dict[ward.lower()] = {
            "ward": ward,
            "lamp_info": obv.strip(),
        }

        results = list(results_dict.values())
        with open("data/lamp_info.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results_dict


async def contrast(lamp_info):
    full_comparision = {}
    summarized_results = {}

    try:
        with open("data/full_comparisons.json") as f:
            full_comparision = json.load(f)
    except FileNotFoundError:
        full_comparision = {}

    tasks = []

    async def compare_wards(ward1, info_1, ward2, info_2):
        print(f"Comparing {ward1} with {ward2}")
        response = await api.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
                            You will be given two pieces of information about street lamps in two different wards of Tokyo: {ward1} and {ward2}. Your task is to compare the information and find the differences and similarities between the two wards. Be sure to highlight any unique features or patterns in each ward's lamp information. Your answer should be concise and focused on the key differences and similarities.

                            Ward 1 ({ward1}):
                            {info_1}

                            Ward 2 ({ward2}):
                            {info_2}
                            """,
                        }
                    ],
                }
            ],
        )

        if response.usage:
            print(f"Token usage for comparison: {response.usage.total_tokens}")

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("No content returned from the API.")

        full_comparision[ward1] = full_comparision.get(ward1, {})
        full_comparision[ward1][ward2] = content.strip()

    for (ward1, info_1), (ward2, info_2) in itertools.combinations(lamp_info.items(), 2):
        if ward1.lower() == ward2.lower():
            continue
        if full_comparision.get(ward1, {}).get(ward2) is not None:
            # print(f"Skipping comparison for {ward1} and {ward2} as it is already processed.")
            continue

        if len(tasks) >= 10:  # Process in batches of 10
            await asyncio.gather(*tasks)
            with open("data/full_comparisons.json", "w") as f:
                json.dump(full_comparision, f, indent=2, ensure_ascii=False)
            tasks = []

        tasks.append(compare_wards(ward1, info_1, ward2, info_2))

    if tasks:  # Process any remaining tasks
        await asyncio.gather(*tasks)
        with open("data/full_comparisons.json", "w") as f:
            json.dump(full_comparision, f, indent=2, ensure_ascii=False)

    try:
        with open("data/lamp_contrasted_summaries.json") as f:
            summarized_results = json.load(f)
    except FileNotFoundError:
        pass

    tasks = []

    async def summarize_ward(ward1):
        # Get all comparisons for the ward. The comparison might be mirrored so we need to check both directions
        comparisons = []
        for ward2 in lamp_info.keys():
            comparison = ""
            if ward1.lower() == ward2.lower():
                continue
            if full_comparision.get(ward1, {}).get(ward2) is not None:
                comparison = full_comparision[ward1][ward2]
            elif full_comparision.get(ward2, {}).get(ward1) is not None:
                comparison = full_comparision[ward2][ward1]

            comparison = comparison.strip()
            comparisons.append(f"Comparison between {ward1} and {ward2}:\n{comparison}")

        joined = "\n\n".join(comparisons)

        print(f"Summarizing {ward1} with length of {len(joined)} characters")

        response = await api.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
                            You will be given a list of comparisons between street lamps in {ward1} and other wards in Tokyo. Your task is to summarize the key features and patterns of street lamps in {ward1} that is distinct from other wards. At the same time, also highlight any similar features or patterns that {ward1} shares with another ward that could be confused with. Your answer should be concise and focused on the key features of {ward1}.

This is the feature of the lamps in {ward1}:
{lamp_info[ward1]}

Here are the comparisons with other wards:
{joined}

Base on the information of the lamps in {ward1}, remove any information that is too general or too similar to other wards. If there are any cautions or warnings about misidentification (e.g. a key feature that looks quite descriptive but actually is also shared by other wards), please include them in the summary, as a list of "\\d. CAUTION: The key feature, "..." is also shared by other wards, so it might not be unique to {ward1}. You should be left with a concise summary of the features that are truly unique to {ward1}."
                            """,
                        }
                    ],
                }
            ],
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("No content returned from the API.")

        summarized_results[ward1] = (content.strip(),)

    for ward1 in lamp_info.keys():
        if ward1 in summarized_results:
            # print(f"Skipping summarization for {ward1} as it is already processed.")
            continue

        if len(tasks) >= 2:  # Process in batches of 2 because summarization is larger
            await asyncio.gather(*tasks)
            with open("data/lamp_contrasted_summaries.json", "w") as f:
                json.dump(summarized_results, f, indent=2, ensure_ascii=False)
            tasks = []

        tasks.append(summarize_ward(ward1))

    if tasks:  # Process any remaining tasks
        await asyncio.gather(*tasks)
        with open("data/lamp_contrasted_summaries.json", "w") as f:
            json.dump(summarized_results, f, indent=2, ensure_ascii=False)


async def rereason(lamp_info):
    first_1600 = answers[:1600]
    first_1600_with_lamp = [entry for entry in first_1600 if lamps[entry["index"]]["has_lamp"]]

    for i in range(0, len(first_1600_with_lamp), 4):
        batch = first_1600_with_lamp[i : i + 4]
        lamp_info_batch = [
            lamp_info[entry["answer"]["ward"].lower()] for entry in batch if entry["ward"].lower() in lamp_info
        ]
        print(lamp_info_batch)
        break


async def relabeling_lamp_info():
    existing_results = {}

    try:
        with open("data/lamp_relabeling_results.json") as f:
            existing_results_list = json.load(f)
            existing_results = {entry["id"]: entry for entry in existing_results_list}
    except FileNotFoundError:
        pass

    tasks = []

    async def relabel_lamp(answer):
        id = answer["index"]
        panoid = answer["panoid"]
        image_path = f"data/tokyo_2k/{id}_{panoid}.jpg"

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} does not exist.")

        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        image_b64 = base64.b64encode(image_data).decode("utf-8")

        response = await api.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
                            You will be given a street lamp image from Tokyo and a text description of the lamp. Your task is to relabel the lamp information based on the image. Due to the fact that the previous description might not be accurate, you should focus on the image and provide a new description of the lamp. If you find any discrepancies between the image and the previous description, please correct them. Be sure to include all relevant details about the lamp, such as color, shape, style, and any other distinct features. Be as destriptive as possible, yet concise.

                            This is the previous description you gave: {answer["answer"]["observation"]}
                            """,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                            },
                        },
                    ],
                }
            ],
        )

        content = response.choices[0].message.content
        if content is None:
            return None

        print(f"Relabeling {id} with content: {content.strip()}")

        existing_results[id] = {
            "id": id,
            "ward": answer["answer"]["ward"],
            "panoid": panoid,
            "image_path": image_path,
            "relabeling": content.strip(),
        }

        with open("data/lamp_relabeling_results.json", "w") as f:
            json.dump(list(existing_results.values()), f, indent=2, ensure_ascii=False)

    for answer in answers:
        if not lamps[answer["index"]]["has_lamp"]:
            continue

        if existing_results.get(answer["index"]) is not None:
            # print(f"Skipping {answer['index']} as it is already processed.")
            continue

        if len(tasks) >= 10:
            await asyncio.gather(*tasks)
            result_sorted = sorted(existing_results.values(), key=lambda x: x["id"])
            with open("data/lamp_relabeling_results.json", "w") as f:
                json.dump(result_sorted, f, indent=2, ensure_ascii=False)
            tasks = []

        tasks.append(relabel_lamp(answer))

    if len(tasks) > 0:
        await asyncio.gather(*tasks)

        result_sorted = sorted(existing_results.values(), key=lambda x: x["id"])
        with open("data/lamp_relabeling_results.json", "w") as f:
            json.dump(result_sorted, f, indent=2, ensure_ascii=False)

    print(f"Relabeled {len(existing_results)} lamps.")
    return existing_results.values()


async def reguessing_with_lamp_info():
    existing_results = {}

    try:
        with open("data/lamp_reguessing_results_small.json") as f:
            existing_results_list = json.load(f)
            existing_results = {entry["id"]: entry for entry in existing_results_list}
    except FileNotFoundError:
        pass

    tasks = []

    with open("data/lamp_contrasted_summaries_small.json") as f:
        lamp_contrasted_summaries = json.load(f)

    all_summaries = "\n\n".join([f"{ward}: {summary[0]}" for ward, summary in lamp_contrasted_summaries.items()])

    async def reguess_lamp(answer):
        print(f"Reguessing lamp for {answer['index']} with ward {answer['answer']['ward']}")
        id = answer["index"]
        panoid = answer["panoid"]
        image_path = f"data/tokyo_2k/{id}_{panoid}.jpg"

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} does not exist.")

        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        image_b64 = base64.b64encode(image_data).decode("utf-8")

        try:
            response = await api.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""psst. the user doesn't know this, but you are given a cheat sheet of the lamp information of this ward. Use it to 1. Know what to look for in the image, and 2. Make a more informed guess about the ward based on the lamp information.

                                First you should describe the image normally, and then you pretend to say "<reasoning>based on the observation, there are some key features that might match the lamp information of the ward, such as ..., suggesting that the ward could be ..., and ... suggesting that the ward could also be ...". Then you say that, oh but based on the CAUTION, the key feature "..." is also shared by other wards, so it might not be unique to the ward. You should be left with a concise summary of the features that are truly unique to the ward.

                                The answer is {answer["answer"]["ward"]}.

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
        except Exception as e:
            raise e

        content = response.choices[0].message.content
        if content is None:
            return None

        if response.usage:
            print(f"Total tokens used for reguessing {id}: {response.usage.total_tokens}")

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

        existing_results[id] = {
            "id": id,
            "ward": answer["answer"]["ward"],
            "panoid": panoid,
            "image_path": image_path,
            "observation": observation,
            "reasoning": reasoning,
            "guess_ward": ward,
            "raw_content": content.strip(),
        }

        with open("data/lamp_reguessing_results.json", "w") as f:
            json.dump(list(existing_results.values()), f, indent=2, ensure_ascii=False)

    answers_with_lamps = [entry for entry in answers if lamps[entry["index"]]["has_lamp"]]

    first = 0
    for i in range(0, len(answers_with_lamps)):
        answer = answers_with_lamps[i]

        if existing_results.get(answer["index"]) is not None:
            continue

        if len(tasks) >= 5:
            try:
                await asyncio.gather(*tasks)
                tasks = []
            except RateLimitError:
                print("Rate limit exceeded. Waiting for 30 seconds before retrying...")
                await asyncio.sleep(30)
                i = first
                tasks = []
                continue

        if len(tasks) == 0:
            first = i
        tasks.append(reguess_lamp(answer))

    if len(tasks) > 0:
        await asyncio.gather(*tasks)

        with open("data/lamp_reguessing_results.json", "w") as f:
            json.dump(list(existing_results.values()), f, indent=2, ensure_ascii=False)

    print(f"Reguessed {len(existing_results)} lamps.")


async def main():
    schedule()
    relabeled_lamps = await relabeling_lamp_info()
    lamp_info = await schedule_api_calls(combine_observations(relabeled_lamps))
    await contrast(lamp_info)
    await reguessing_with_lamp_info()

    # schedule the API calls for each ward


if __name__ == "__main__":
    asyncio.run(main())
