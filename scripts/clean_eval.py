import argparse
import json


def clean_eval_file(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    with open("../output_gpt.json", "r") as f:
        gpt_data = json.load(f)["results"]

    cleaned_data = []
    for entry in data:
        if "entry" in entry:
            entry_data = entry["entry"]
            cleaned_entry = {
                **entry,
                "match": entry_data["ward"] == gpt_data[entry["id"]]["answer"]["ward"],
            }
            cleaned_data.append(cleaned_entry)

    with open(output_file, "w") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Clean evaluation results.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input evaluation file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the cleaned evaluation file.")

    args = parser.parse_args()

    clean_eval_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
