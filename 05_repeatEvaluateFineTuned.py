import transformers
import torch

model_id = "./fine_tuned_llama"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

pipeline.model.eval()

PROMPT = '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.'''

print("AI Assistant와 대화를 시작합니다. '종료'라고 입력하면 종료됩니다.")
while True:
    instruction = input("User: ")
    if instruction.lower() in ['종료', 'exit', 'quit']:
        print("대화를 종료합니다.")
        break

    full_prompt = f"{PROMPT}\nUser: {instruction}\nAI:"
    terminators = [pipeline.tokenizer.eos_token_id]

    outputs = pipeline(
        full_prompt,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )

    generated_text = outputs[0]["generated_text"]
    response = generated_text[len(full_prompt):]

    stop_phrases = ["User:", "AI:", "\n", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."]

    for stop_phrase in stop_phrases:
        if stop_phrase in response:
            response = response.split(stop_phrase)[0].strip()

    print(f"AI: {response}")