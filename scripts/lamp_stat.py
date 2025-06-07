import json


with open("data/has_lamp_results.json") as f:
    lamps = json.load(f)


def main():
    # with see the distribution of cities that have lamps and those that do not
    total_lamps = 0
    processed = 0
    cities_lamp_count = {}

    for entry in lamps:
        if "ward" not in entry or "has_lamp" not in entry:
            print(f"Skipping entry {entry} due to missing fields.")
            continue

        city = entry["ward"].lower()
        processed += 1
        has_lamp = entry["has_lamp"]
        total_lamps += 1 if has_lamp else 0

        cities_lamp_count[city] = cities_lamp_count.get(city, 0) + (1 if has_lamp else 0)

    print("City Lamp Counts:")
    for city, count in cities_lamp_count.items():
        if count > 0:
            print(f"{city}: {count} lamps")

    print(f"Total lamps: {total_lamps} / {processed} processed entries")


if __name__ == "__main__":
    main()
