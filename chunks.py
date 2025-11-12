class Chunks:
    def __init__(self, chunk_size=5000, overlap_size=1000):

        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        
        # Validação básica
        if self.overlap_size >= self.chunk_size:
            raise ValueError("O tamanho do overlay deve ser menor que o tamanho do chunk")
        if self.chunk_size <= 0 or self.overlap_size < 0:
            raise ValueError("Tamanhos devem ser valores positivos")
    
    def create_chunks(self, text):

        if not text or not isinstance(text, str):
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Define o final do chunk atual
            end = start + self.chunk_size
            
            # Se não é o último chunk, tenta quebrar em uma posição melhor
            if end < text_length:
                # Procura por quebras naturais próximas ao final do chunk
                chunk_text = text[start:end]
                
                # Tenta quebrar em parágrafo (dupla quebra de linha)
                last_paragraph = chunk_text.rfind('\n\n')
                if last_paragraph > len(chunk_text) * 0.7:  # Se encontrou um parágrafo nos últimos 30%
                    end = start + last_paragraph + 2
                
                # Se não encontrou parágrafo, tenta quebrar em frase (ponto + espaço)
                elif '. ' in chunk_text:
                    last_sentence = chunk_text.rfind('. ')
                    if last_sentence > len(chunk_text) * 0.7:  # Se encontrou uma frase nos últimos 30%
                        end = start + last_sentence + 2
                
                # Se não encontrou frase, tenta quebrar em palavra (espaço)
                elif ' ' in chunk_text:
                    last_space = chunk_text.rfind(' ')
                    if last_space > len(chunk_text) * 0.8:  # Se encontrou um espaço nos últimos 20%
                        end = start + last_space + 1
            
            # Extrai o chunk atual
            chunk = text[start:end].strip()
            if chunk:  # Só adiciona se o chunk não estiver vazio
                chunks.append(chunk)
            
            # Calcula a próxima posição inicial considerando o overlay
            if end >= text_length:
                break
            
            start = end - self.overlap_size
            
            # Garante que não voltamos para trás demais
            if start < 0:
                start = 0
        
        return chunks
    
    def create_chunks_with_metadata(self, text, source_info=None):
        chunks = self.create_chunks(text)
        
        chunks_with_metadata = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                'chunk_id': i,
                'chunk_text': chunk,
                'chunk_size': len(chunk),
                'total_chunks': len(chunks),
                'chunk_start_char': self._calculate_start_position(chunks, i),
                'source_info': source_info or {}
            }
            chunks_with_metadata.append(chunk_metadata)
        
        return chunks_with_metadata
    
    def _calculate_start_position(self, chunks, chunk_index):
        if chunk_index == 0:
            return 0
        
        # Estimativa baseada no tamanho dos chunks anteriores menos os overlays
        estimated_position = 0
        for i in range(chunk_index):
            estimated_position += len(chunks[i]) - self.overlap_size
        
        return max(0, estimated_position)
    
    def get_chunk_info(self):
        
        return {
            'chunk_size': self.chunk_size,
            'overlap_size': self.overlap_size,
            'effective_chunk_step': self.chunk_size - self.overlap_size
        }
    
    def update_settings(self, chunk_size=None, overlap_size=None):

        if chunk_size is not None:
            self.chunk_size = chunk_size
        
        if overlap_size is not None:
            self.overlap_size = overlap_size
        
        # Revalida após a atualização
        if self.overlap_size >= self.chunk_size:
            raise ValueError("O tamanho do overlay deve ser menor que o tamanho do chunk")
        if self.chunk_size <= 0 or self.overlap_size < 0:
            raise ValueError("Tamanhos devem ser valores positivos")
