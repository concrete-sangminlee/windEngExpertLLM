import json
import tiktoken  # OpenAI의 tiktoken 라이브러리


# JSONL 파일의 각 텍스트에 대해 토큰 수를 계산하는 함수
def count_tokens_in_jsonl(filename='fine_tuning_data.jsonl', model_name='gpt-3.5-turbo'):
    try:
        # OpenAI 모델에 맞는 토크나이저를 로드
        enc = tiktoken.encoding_for_model(model_name)

        total_tokens = 0
        # JSONL 파일 읽기
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                # 각 줄을 JSON 객체로 파싱
                data = json.loads(line.strip())
                prompt = data.get("prompt", "")
                completion = data.get("completion", "")

                # Prompt와 Completion에 대한 토큰 계산
                prompt_tokens = len(enc.encode(prompt))
                completion_tokens = len(enc.encode(completion))
                total_tokens += prompt_tokens + completion_tokens

                print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")

        # 총 토큰 수 출력
        print(f"Total tokens in file: {total_tokens}")
        return total_tokens

    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON. Please check the file format.")


# 메인 함수 실행
if __name__ == "__main__":
    count_tokens_in_jsonl()  # 기본 파일 이름을 사용하지만 다른 파일명을 사용할 수 있음