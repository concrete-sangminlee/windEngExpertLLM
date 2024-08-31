import pandas as pd
import json

excel_file = ('customData.xlsx')

df = pd.read_excel(excel_file)

jsonl_file = 'customData.jsonl'

with open(jsonl_file, 'w', encoding='utf-8') as f:
    for index, row in df.iterrows():
        entry = {
            "messages": [
                {"role": "system", "content": row['system_content']},
                {"role": "user", "content": row['user_content']},
                {"role": "assistant", "content": row['assistant_content']}
            ]
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"JSONL 파일로 변환 완료: {jsonl_file}")