import json


# JSONL 파일에 데이터를 저장하는 함수
def save_to_jsonl(input_text, output_text, filename='fine_tuning_data.jsonl'):
    # 입력과 출력을 JSON 형식으로 변환
    data = {
        "prompt": input_text.strip(),
        "completion": output_text.strip()
    }

    # 파일에 추가 모드('a')로 열어서 데이터를 저장
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"Data saved to {filename}")


# 반복적으로 입력을 받는 루프
def input_loop():
    print("Enter text inputs for fine-tuning. Type 'exit' to stop.")

    while True:
        # 사용자 입력 받기
        input_text = input("Enter your prompt (input text): ")
        if input_text.lower() == 'exit':
            print("Exiting...")
            break

        output_text = input("Enter the expected completion (output text): ")
        if output_text.lower() == 'exit':
            print("Exiting...")
            break

        # JSONL 파일에 저장
        save_to_jsonl(input_text, output_text)


# 메인 루프 실행
if __name__ == "__main__":
    input_loop()
