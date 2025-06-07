import asyncio
import json
import itertools
import os
import re
import sys

from openai import AsyncOpenAI

api = AsyncOpenAI(api_key="0", base_url="http://0.0.0.0:8000/v1")


with open("data/has_lamp_results.json") as f:
    lamps = json.load(f)

with open("../output_gpt.json") as f:
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


def combine_observations():
    combined_results = []
    for ward, entries in wards.items():
        combined_observation = "\n\n".join(answers[entry["id"]]["answer"]["observation"] for entry in entries)

        combined_results.append(
            {
                "ward": ward,
                "observation": combined_observation,
            }
        )

    # with open("data/combined_observations.json", "w") as f:
    #     json.dump(combined_results, f, indent=4)

    return combined_results


async def get_lamp_info(ward: str, observation: str):
    response = await api.chat.completions.create(
        model="Qwen/Qwen3-8B",
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

    answer = re.sub(r"<think>[\s\S]*?</think>", "", content).strip().lower()
    return ward, answer


async def schedule_api_calls(combined_results):
    existing_wards = set()
    results_dict = {}

    try:
        with open("data/combined_lamp_info.json") as f:
            results_dict = json.load(f)
            existing_wards = {entry[0].lower() for entry in results_dict}
    except FileNotFoundError:
        pass

    # if existing_wards has all entries for ward in combined_results, skip the entire processing
    if all(entry["ward"].lower() in existing_wards for entry in combined_results):
        print("All wards already processed. Skipping API calls.")
        return results_dict

    tasks = []
    for entry in combined_results:
        ward = entry["ward"]
        observation = entry["observation"]
        tasks.append(get_lamp_info(ward, observation))

    results = asyncio.run(asyncio.gather(*tasks))

    results_dict = {}  # ward -> lamp_info
    for ward, lamp_info in results:
        results_dict[ward] = lamp_info
    results = list(results_dict.items())
    results.sort(key=lambda x: x[0])
    with open("data/lamp_info.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results_dict


async def contrast(lamp_info):
    full_comparision = []
    summarized_results = []
    for ward1, info_1 in lamp_info.items():
        comparison_results = []
        for ward2, info_2 in lamp_info.items():
            if ward1 == ward2:
                continue

            print(f"Comparing {ward1} with {ward2}")
            response = await api.chat.completions.create(
                model="Qwen/Qwen3-8B",
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

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("No content returned from the API.")
            comparison_results.append(
                {
                    "ward1": ward1,
                    "ward2": ward2,
                    "comparison": content.strip(),
                }
            )

        full_comparision.extend(comparison_results)
        joined = "\n\n".join([f"{entry['ward2']}: {entry['comparison']}" for entry in comparison_results])
        response = await api.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
                            You will be given a list of comparisons between street lamps in {ward1} and other wards in Tokyo. Your task is to summarize the key features and patterns of street lamps in {ward1} that is distinct from other wards. At the same time, also highlight any similar features or patterns that {ward1} shares with another ward that could be confused with. Your answer should be concise and focused on the key features of {ward1}.

                            Comparisons for {ward1}:
                            {joined}
                            """,
                        }
                    ],
                }
            ],
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("No content returned from the API.")

        summarized_results.append(
            {
                "ward": ward1,
                "summary": content.strip(),
            }
        )

    with open("data/lamp_comparisons.json", "w") as f:
        json.dump(full_comparision, f, indent=2, ensure_ascii=False)

    with open("data/lamp_summaries.json", "w") as f:
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


if __name__ == "__main__":
    schedule()
    combined_results = combine_observations()
    lamp_info = asyncio.run(schedule_api_calls(combined_results))
    # asyncio.run(contrast(lamp_info))

    # schedule the API calls for each ward
