import json

# JSONL 파일에 데이터를 저장하는 함수
def save_to_jsonl(input_text, output_text, filename='fine_tuning_data.jsonl'):
    data = {
        "prompt": input_text.strip(),
        "completion": output_text.strip()
    }

    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"Data saved to {filename}")

# 최근에 추가된 JSONL 항목을 삭제하는 함수
def delete_last_entry(filename='fine_tuning_data.jsonl'):
    try:
        # 파일을 읽어서 모든 줄을 리스트로 저장
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 파일에 적어도 한 줄이 있을 경우 마지막 항목을 삭제
        if lines:
            lines = lines[:-1]  # 마지막 항목 삭제

            # 파일을 다시 열고 삭제된 내용을 저장
            with open(filename, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print("Last entry deleted.")
        else:
            print("The file is empty, no entry to delete.")

    except FileNotFoundError:
        print(f"File '{filename}' not found.")

def input_loop():
    print("Enter text inputs for fine-tuning. Type 'exit' to stop.")
    print("Type 'delete' to remove the most recent entry.")

    while True:
        input_text = input("Enter your prompt (input text): ")
        if input_text.lower() == 'exit':
            print("Exiting...")
            break

        if input_text.lower() == 'delete':
            delete_last_entry()
            continue

        output_text = input("Enter the expected completion (output text): ")
        if output_text.lower() == 'exit':
            print("Exiting...")
            break

        save_to_jsonl(input_text, output_text)

if __name__ == "__main__":
    input_loop()
