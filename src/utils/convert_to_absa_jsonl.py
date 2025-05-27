import json
import os

def convert_to_absa_jsonl(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            record = json.loads(line)
            if "data" not in record or "label" not in record:
                continue

            text = record["data"]

            new_item = {
                "text": text,
                "labels": []
            }
            for start, end, full_label in record["label"]:
                if "#" in full_label:
                    sentiment = full_label.split("#")[-1].upper()
                else:
                    sentiment = full_label.upper()
                new_item["labels"].append([start, end, sentiment])
            f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")

    print(f"Đã chuyển xong: {output_path}")

if __name__ == "__main__":
    print("\n\nChuyển đổi định dạng dữ liệu sang ABSA JSONL...")

    input_path = "data/origin/hotel.jsonl"
    output_path = "data/processed/hotel.jsonl"
    convert_to_absa_jsonl(input_path, output_path)