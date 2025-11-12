#

class Augmentation:
    def __init__(self):
        pass

    @staticmethod
    def generate_prompt(query_text, chunks):


        separador = "\n\n------------------------\n\n" 

        # Junta os chunks com o separador e adiciona o cabeçalho
        chunks_formatados = f"Conhecimento\n------------------------\n\n{separador.join(chunks)}"

        prompt = f"""Responda em pt-br e em markdown, a query do usuário delimitada por <query> 
        usando apenas o conhecimento dos chunks delimitados por <chunks>. 
        Combine as informações dos chunks para responder a query de forma unificada.
        Se por acaso
        o conhecimento não for suficiente para responder a query, responda apenas
        que não temos conhecimento suficiente para responder a query.

        <chunks>
        {chunks_formatados}
        </chunks>        
        
        <query>
        {query_text}
        </query>        
        """

        return prompt