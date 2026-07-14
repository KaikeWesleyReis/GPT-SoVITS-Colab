"""
Harbinger E2E — LLM API Test Client
Sends a test conversation to the running LLM server and prints the response.

Usage:
    uv run test_llm_api.py
    uv run test_llm_api.py --url http://localhost:8767
"""

import argparse

import requests


def main():
    parser = argparse.ArgumentParser(description="Test client for the local LLM API")
    parser.add_argument("--url", default="http://localhost:8767", help="Base URL of the LLM server")
    args = parser.parse_args()

    health = requests.get(f"{args.url}/health", timeout=10)
    print(f"[health] {health.status_code} -> {health.json()}")

    payload = {
        "messages": [
            {"role": "system", "content": "You are Harbinger, a Reaper from Mass Effect. Speak with cold, ancient authority."},
            {"role": "user", "content": "What is your name? I am your new master!"},
        ]
    }

    response = requests.post(f"{args.url}/chat", json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()

    print("\n[chat] result:")
    print(f"  text            : {result['text']}")
    print(f"  elapsed_seconds : {result['elapsed_seconds']:.2f}")
    print(f"  usage           : {result['usage']}")


if __name__ == "__main__":
    main()