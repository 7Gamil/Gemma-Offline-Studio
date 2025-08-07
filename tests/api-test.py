from openai import OpenAI
import requests
import os

# Test server health
def test_server_health():
    try:
        response = requests.get("http://localhost:8000")
        print("Server health:", response.json())
    except Exception as e:
        print(f"Server health check failed: {e}")

# Test available models
def test_models():
    try:
        response = requests.get("http://localhost:8000/v1/models")
        print("Available models:", response.json())
    except Exception as e:
        print(f"Models check failed: {e}")

# Initialize OpenAI client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"
)

use_model_E2B = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"
use_model_E4B = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit"

def test_chat():
    try:
        # Test text completion
        response = client.chat.completions.create(
            model=use_model_E2B,
            messages=[
                {"role": "user", "content": "Write a hello world program in Python"}
            ]
        )
        print("\nText completion response:")
        print(response.choices[0].message.content)

        # Test image + text
        response = client.chat.completions.create(
            model=use_model_E2B,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "1.jpg"
                    },
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    }
                ]
            }]
        )
        print("\nImage + text response:")
        print(response.choices[0].message.content)

        # Test audio + image + text
        audio_path = os.path.join(os.path.dirname(__file__), "audio.mp3")
        response = client.chat.completions.create(
            model=use_model_E2B,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": audio_path
                    },
                    {
                        "type": "image",
                        "image": "1.jpg"
                    },
                    {
                        "type": "text",
                        "text": "What is this audio and image about?"
                    }
                ]
            }]
        )
        print("\nAudio + Image + text response:")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"Chat test failed: {e}")

if __name__ == "__main__":
    print("Starting API tests...")
    test_server_health()
    test_models()
    test_chat()