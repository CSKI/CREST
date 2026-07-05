from openai import OpenAI
client = OpenAI(api_key="sk-d6a737a07d9f4a0ebf7ea63997358483", base_url="https://api.deepseek.com")

# Round 1
messages = [{"role": "user", "content": "What's the highest mountain in the world?"}]
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages
)
messages.append(response.choices[0].message)
#messages.append(response.choices[0].message)
print(f"Messages Round 1: {messages}")

# Round 2
messages.append({"role": "user", "content": "What is the second?"})
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages
)

# messages.append(response.choices[0].message)
print(f"Messages Round 2: {response.choices[0].message}")