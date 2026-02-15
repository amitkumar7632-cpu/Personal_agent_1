from google import genai

client = genai.Client(api_key="AIzaSyAqXiJcrigh5bxeiWPnspnLcOWyCRcGqVE")

for m in client.models.list():
    print(m.name)