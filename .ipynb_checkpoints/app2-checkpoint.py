"""
test_gemini.py -- quick sanity check for Gemini API setup

Run:
    pip install google-genai
    setx GEMINI_API_KEY "your-key-here"   (Windows -- open a NEW terminal after this)
    python test_gemini.py
"""

import os
from google import genai

def main():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("❌ GEMINI_API_KEY is not set in this terminal session.")
        print("   Run: setx GEMINI_API_KEY \"your-key-here\"")
        print("   Then open a NEW terminal window and try again.")
        return

    print(f"✅ Found API key (starts with: {api_key[:6]}...)")

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents="Reply with exactly one word: pong"
        )

        print("✅ API call succeeded.")
        print("Response text:", response.text.strip())

    except Exception as e:
        print("❌ API call failed.")
        print("Error:", e)

if __name__ == "__main__":
    main()