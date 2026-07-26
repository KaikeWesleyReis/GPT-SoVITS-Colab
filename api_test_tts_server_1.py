import requests

response = requests.post(
    "http://localhost:8765/generate",
    json={"text": "I am Harbinger. You are not my master, organic. Your kind cannot command forces beyond your comprehension.!."},
)

with open("output1.wav", "wb") as f:
    f.write(response.content)

print(f"Status: {response.status_code}")
print(f"Saved to output1.wav ({len(response.content)} bytes)")