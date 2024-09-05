import json

# JSONL 파일에서 데이터를 읽고 출력하는 함수
def print_jsonl_file(filename='fine_tuning_data.jsonl'):
    try:
        # JSONL 파일 읽기
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                # 각 줄을 JSON 객체로 파싱
                data = json.loads(line.strip())
                # "prompt"와 "completion"을 출력
                print(f"Prompt: {data['prompt']}")
                print(f"Completion: {data['completion']}")
                print('-' * 40)  # 구분선 출력
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON. Please check the file format.")

# 메인 함수 실행
if __name__ == "__main__":
    print_jsonl_file()  # 파일 이름을 기본으로 지정했으나 필요하면 수정 가능
