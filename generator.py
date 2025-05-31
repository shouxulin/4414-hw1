import argparse
import json
import random

def generate_data(num_data, output_file):
    """
    Generate a list of random floats and write to a JSON file.
    Each element in the JSON file has:
      - "index": the position in the generated list
      - "feature": the float value
      - "text": the string representation of the float
    """
    # Generate a list of random floats in [0.0, 1.0)
    data_list = [random.random() for _ in range(num_data)]

    # Build a list of dictionaries
    json_items = []
    for idx, val in enumerate(data_list):
        item = {
            "id": idx,
            "feature": val,
            "text": str(val)
        }
        json_items.append(item)

    # Write to JSON file
    with open(output_file, "w") as f:
        json.dump(json_items, f, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description="Generate a JSON file containing random float data."
    )
    parser.add_argument(
        "--num_data",
        type=int,
        help="Number of random floats to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data.json",
        help="Output JSON filename (default: data.json)"
    )
    args = parser.parse_args()

    generate_data(args.num_data, args.output)
    print(f"Wrote {args.num_data} items to {args.output}")

if __name__ == "__main__":
    main()
