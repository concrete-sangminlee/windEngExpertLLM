import transformers
import torch
from datetime import datetime

model_id = "./fine_tuned_llama_B"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

PROMPT = '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.'''

print("AI Assistant와 대화를 시작합니다. '종료'라고 입력하면 종료됩니다.")

conversation_history = []


def get_full_prompt(instruction):
    global conversation_history
    conversation_history.append(f"User: {instruction}")

    # 최근 대화 몇 개만 유지
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]

    return PROMPT + "\n".join(conversation_history) + "\nAI:"


while True:
    instruction = input("User: ")
    if instruction.lower() in ['종료', 'exit', 'quit']:
        print("대화를 종료합니다.")
        break

    full_prompt = get_full_prompt(instruction)
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(full_prompt):].strip()

    # AI 응답 저장
    conversation_history.append(f"AI: {response}")
    print(f"AI: {response}")

    # 대화 기록을 파일에 저장
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_name = datetime.now().strftime("conversation_history_%Y%m%d_%H%M%S.txt")
    with open(file_name, "w", encoding="utf-8") as f:
        f.write("\n".join(conversation_history))

print(f"대화 내역이 '{file_name}' 파일에 저장되었습니다.")