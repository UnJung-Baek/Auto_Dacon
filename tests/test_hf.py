from huggingface_hub import InferenceClient

client = InferenceClient(api_key="")

messages = [
    {
        "role": "user",
        "content": "What is the capital of France"
    }
]

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct",
    messages=messages,
    max_tokens=3000
)

print(completion.choices[0].message.content)
