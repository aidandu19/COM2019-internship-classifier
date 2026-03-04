from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("OPENAI_API_KEY")

print("Loaded:", bool(key))
print("Starts with sk-:", str(key).startswith("sk-"))
print("Length:", len(key) if key else 0)