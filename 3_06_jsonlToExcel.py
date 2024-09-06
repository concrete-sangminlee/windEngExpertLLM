import json
import pandas as pd

def jsonl_to_excel(jsonl_filename='fine_tuning_data.jsonl', excel_filename='fine_tuning_data.xlsx'):
    try:
        data_list = []

        with open(jsonl_filename, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                data_list.append(data)

        df = pd.DataFrame(data_list)

        df.to_excel(excel_filename, index=False)
        print(f"Data successfully saved to {excel_filename}")

    except FileNotFoundError:
        print(f"File '{jsonl_filename}' not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON. Please check the file format.")

if __name__ == "__main__":
    jsonl_to_excel()