import pandas as pd
import json


def augment_data(df):
    augmented_data = []
    for index, row in df.iterrows():
        user_content = row['user_content']
        assistant_content = row['assistant_content']

        # 간단한 동의어 교체를 통한 데이터 증강
        synonyms = {
            "풍하중": ["바람 하중", "풍력 하중"],
            "구조 설계": ["건축 설계", "빌딩 설계"]
        }

        for key, values in synonyms.items():
            for synonym in values:
                new_user_content = user_content.replace(key, synonym)
                new_row = row.copy()
                new_row['user_content'] = new_user_content
                augmented_data.append(new_row)

        augmented_data.append(row)

    return pd.DataFrame(augmented_data)


# 원본 데이터 불러오기
excel_file = 'customData.xlsx'
df = pd.read_excel(excel_file)

# 데이터 증강 수행
augmented_df = augment_data(df)

# 증강된 데이터로 JSONL 변환
jsonl_file = 'customData_augmented.jsonl'
with open(jsonl_file, 'w', encoding='utf-8') as f:
    for index, row in augmented_df.iterrows():
        entry = {
            "messages": [
                {"role": "system", "content": row['system_content']},
                {"role": "user", "content": row['user_content']},
                {"role": "assistant", "content": row['assistant_content']}
            ]
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"증강된 데이터로 JSONL 파일로 변환 완료: {jsonl_file}")