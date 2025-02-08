import base64
import requests
import os
from dotenv import load_dotenv
import expert_prompt

load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"
MAX_TOKENS = 2_000
TEMPERATURE = 0.7

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def make_text_image_request(prompt: str, image_path: str) -> dict:
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()
    return response_data

def read_content(response_data: dict) -> list[str]:
    if "choices" not in response_data:
        return []
    return [choice["message"]["content"] for choice in response_data["choices"]]


if __name__ == "__main__":
    prompt = expert_prompt.get_instructions()
    image_path = "backend/src/ai/files/14_-_Alzaibar_entre_Buenos_Aires_y_Sarandi_hacia_el_Sur.jpg"
    response_data = make_text_image_request(prompt, image_path)
    for i, content in enumerate(read_content(response_data)):
        print(f"{i}\n{content}")
        print("----"*10)
    print("Done")