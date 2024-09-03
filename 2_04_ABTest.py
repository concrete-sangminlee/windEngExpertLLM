import torch
from transformers import pipeline

# 두 개의 미세 조정된 모델 로드
model_A = "./fine_tuned_llama_A"
model_B = "./fine_tuned_llama_B"

pipeline_A = pipeline("text-generation", model=model_A, tokenizer=model_A, model_kwargs={"torch_dtype": torch.bfloat16},
                      device_map="auto")
pipeline_B = pipeline("text-generation", model=model_B, tokenizer=model_B, model_kwargs={"torch_dtype": torch.bfloat16},
                      device_map="auto")

# A/B 테스트 함수
def run_ab_test(user_input):
    response_A = pipeline_A(user_input, max_new_tokens=2048, do_sample=True, temperature=0.7, top_p=0.9)[0][
        "generated_text"]
    response_B = pipeline_B(user_input, max_new_tokens=2048, do_sample=True, temperature=0.7, top_p=0.9)[0][
        "generated_text"]

    print("Model A Response:", response_A)
    print("Model B Response:", response_B)

# 사용자 입력을 받는 루프
while True:
    user_input = input("User: ")
    if user_input.lower() in ['종료', 'exit', 'quit']:
        print("대화를 종료합니다.")
        break

    run_ab_test(user_input)