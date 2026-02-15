from google import genai

# Direct API key (replace with your valid Gemini key from AI Studio)
client = genai.Client(api_key="AIzaSyAqXiJcrigh5bxeiWPnspnLcOWyCRcGqVE")

# Use one of the available models
response = client.models.generate_content(
    model="models/gemini-2.5-flash",   # or "models/gemini-pro-latest"
    contents="Hello Gemini, india capital?"
)

print(response.text)