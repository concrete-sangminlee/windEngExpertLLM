import transformers
import torch

model_id = "./fine_tuned_llama_A"
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
while True:
    instruction = input("User: ")
    if instruction.lower() in ['종료', 'exit', 'quit']:
        print("대화를 종료합니다.")
        break

    inputs = tokenizer(PROMPT + f"\nUser: {instruction}\nAI:", return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = generated_text[len(PROMPT + f"\nUser: {instruction}\nAI:"):].strip()

    stop_phrases = ["User:", "AI:", "\n", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."]
    for stop_phrase in stop_phrases:
        if stop_phrase in response:
            response = response.split(stop_phrase)[0].strip()

    print(f"AI: {response}")