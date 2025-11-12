from google import genai
from dotenv import load_dotenv
import os


class Generation:
    
    def __init__(self, model="gemini-2.5-flash-lite"):
        load_dotenv()
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model
     
    def generate(self, prompt):

        client = genai.Client()

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text
