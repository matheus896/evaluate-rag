import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from markitdown import MarkItDown

from google import genai
from google.genai import types
from dotenv import load_dotenv
import base64

load_dotenv()

# Only run this block for Gemini Developer API
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiWrapper:
    """Wrapper para adaptar Gemini Client à interface OpenAI esperada pelo MarkItDown"""
    def __init__(self, gemini_client):
        self.gemini_client = gemini_client
        self.chat = self
        
    class Completions:
        def __init__(self, gemini_client):
            self.gemini_client = gemini_client
            
        def create(self, model, messages):
            # Extrai o prompt de texto e a imagem do formato OpenAI
            prompt_text = None
            image_data = None
            base64_data = None
            mime_type = "image/png"
            
            for msg in messages:
                if msg.get("role") == "user":
                    for content in msg.get("content", []):
                        if content.get("type") == "text":
                            prompt_text = content.get("text")
                        elif content.get("type") == "image_url":
                            image_url = content.get("image_url", {}).get("url")
                            if image_url and image_url.startswith("data:"):
                                # Extrai mime type e dados base64
                                parts = image_url.split(",")
                                header = parts[0]  # data:image/png;base64
                                base64_data = parts[1]
                                
                                # Extrai o mime type
                                if ":" in header and ";" in header:
                                    mime_type = header.split(":")[1].split(";")[0]
            
            # Prepara o conteúdo no formato correto para Gemini
            parts = []
            if prompt_text:
                parts.append(types.Part(text=prompt_text))
            if base64_data:
                parts.append(types.Part(
                    inline_data=types.Blob(
                        mime_type=mime_type,
                        data=base64.b64decode(base64_data)
                    )
                ))
            
            # Chama a API do Gemini com o formato correto
            response = self.gemini_client.models.generate_content(
                model=model,
                contents=types.Content(parts=parts)
            )
            
            # Retorna no formato esperado (similar ao OpenAI)
            class Choice:
                def __init__(self, text):
                    self.message = type('obj', (object,), {'content': text})()
            
            class Response:
                def __init__(self, text):
                    self.choices = [Choice(text)]
            
            return Response(response.text)
    
    @property
    def completions(self):
        return self.Completions(self.gemini_client)

client = GeminiWrapper(gemini_client)

class ReadFiles:
    def __init__(self):
        pass
    
    def docs_to_markdown(self, dir_path):
        
        png = self.read_dir(dir_path)
        processed_files = []  # Track files processed in this run

        for file in png:
            
            file_path = os.path.join(dir_path, file)
        
            # quero que vc leia o file e diga qual o tipo do arquivo (pdf, doc, docx, image e ect)
            extension = file.split('.')[-1] 
            
            if extension == 'pdf' or \
            extension == 'doc' or \
            extension == 'docx' or \
            extension == "xls" or \
            extension == "xlsx" or \
            extension == "ppt" or \
            extension == "pptx" or \
            extension == "csv" or \
            extension == "txt" or \
            extension == "json" or \
            extension == "xml" or \
            extension == "html" or \
            extension == "htm" or \
            extension == "yaml": 
                
                md = MarkItDown(enable_plugins=True)
                
                result = md.convert(file_path)
                            
            elif extension == 'jpg' or \
                extension == 'png' or \
                extension == 'jpeg' or \
                extension == 'gif' or \
                extension == 'bmp' or \
                extension == 'webp' or \
                extension == 'svg' or \
                extension == 'tiff' or \
                extension == 'ico':
                
                md = MarkItDown(llm_client=client,
                                llm_model="gemini-2.5-flash",
                                llm_prompt="""Em 3 parágrafos, 
                                descreva a imagem detalhadamente em 
                                pt-br""",
                                enable_plugins=True)
                
                result = md.convert(file_path)
                
            else:
                continue  # Skip unsupported file types
                
            # SALVE O RESULTADO EM UM ARQUIVO MD NA PASTA MARKDOWN
            # O nome do arquivo em "file.split" pode conter mais de um ponto
            # por exemplo: 2111.01888v1.pdf
            # entao o nome do arquivo md deve ser: 2111.01888v1.md
            # No código abaixo vc deve pegar no split o último ponto
            # e usar ele para criar o nome do arquivo md
            # por exemplo: 2111.01888v1.pdf
            # o último ponto é o ".pdf"
            # entao o nome do arquivo md deve ser: 2111.01888v1.md
            

            # Remove apenas a última extensão (após o último ponto)
            filename_without_ext = os.path.splitext(file)[0]
            markdown_dir = os.path.join(os.path.dirname(__file__), "markdown")
            md_path = os.path.join(markdown_dir, filename_without_ext + ".md")
            
            # se não existir a pasta markdown, crie
            if not os.path.exists(markdown_dir):
                os.makedirs(markdown_dir)
            
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(result.text_content)
            
            processed_files.append(filename_without_ext + ".md")
            
        # LER APENAS O CONTEUDO DOS ARQUIVOS PROCESSADOS NESTA EXECUÇÃO
        md_content = ""
        
        markdown_dir = os.path.join(os.path.dirname(__file__), "markdown")
        for md_file in processed_files:
            with open(os.path.join(markdown_dir, md_file), "r", encoding="utf-8") as f:
                md_content += f"\n\n{'='*80}\n"
                md_content += f"ARQUIVO: {md_file}\n"
                md_content += f"{'='*80}\n\n"
                md_content += f.read()
        
        return md_content
    
    def read_dir(self, dir_path):
        
        files = os.listdir(dir_path)
        
        return files
        
# Script para teste

if __name__ == "__main__":
    reader = ReadFiles()
    docs_path = os.path.join(os.path.dirname(__file__), "png")
    content = reader.docs_to_markdown(docs_path)
    print(content)