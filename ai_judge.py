import os
import json
import time
import logging
from typing import Dict, Any, Optional, cast
from dotenv import load_dotenv

try:
    from litellm import completion
    import litellm
except Exception:
    raise

# Define OpenAIError fallback for error handling
class OpenAIError(Exception):
    pass

load_dotenv()

logger = logging.getLogger(__name__)


class AIJudge:
    """AI Judge wrapper using LiteLLM (litellm.completion).

    Behavior mirrors the previous judge in `evaluate_light.py` but uses
    LiteLLM as the provider-agnostic client. If `model` is not provided in
    calls, the class will attempt to read `test_config.json` for
    `ai_judge_model` and `ai_judge_temperature`.
    """

    def __init__(self, config_path: str = "test_config.json") -> None:
        # Try load config for defaults
        self.config = {}
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            cfg_path = config_path if os.path.isabs(config_path) else os.path.join(script_dir, config_path)
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
                logger.info(f"AIJudge: loaded config from {cfg_path}")
        except Exception:
            # non-fatal; we'll require model to be passed later if missing
            logger.debug("AIJudge: no config file loaded, defaults will be used from args/env")

    def _default_model(self) -> Optional[str]:
        return self.config.get("ai_judge_model") or os.getenv("AI_JUDGE_MODEL")

    def _default_temperature(self) -> float:
        return float(self.config.get("ai_judge_temperature", 0.1) or os.getenv("AI_JUDGE_TEMPERATURE", 0.1))

    def create_judge_prompt(self, question: str, context: str, answer: str) -> str:
        """Create the same judge prompt used previously in evaluate_light.py."""
        return f"""Você é um avaliador especialista em sistemas RAG no domínio jurídico.

Avalie a qualidade da resposta segundo os critérios abaixo, usando escala de 1 a 5:

**PERGUNTA:**
{question}

**CONTEXTO RECUPERADO (5 CHUNKS):**
{context}

**RESPOSTA GERADA:**
{answer}

**CRITÉRIOS (1-5):**
1. **Consistência Factual:** Resposta baseada no contexto? Há alucinações?
2. **Seguir Instruções:** Responde o que foi pedido?
3. **Conhecimento Jurídico:** Usa terminologia correta?
4. **Precisão do Contexto:** Contexto é relevante?
5. **Cobertura do Contexto:** Contexto tem todas as informações necessárias?

**RESPONDA EM JSON:**
{{
  "factual_consistency": {{"score": <1-5>, "justification": "..."}},
  "instruction_following": {{"score": <1-5>, "justification": "..."}},
  "domain_knowledge": {{"score": <1-5>, "justification": "..."}},
  "context_precision": {{"score": <1-5>, "justification": "..."}},
  "context_recall": {{"score": <1-5>, "justification": "..."}},
  "overall_assessment": "resumo geral em 1-2 frases"
}}"""

    def judge_response(
        self,
        question: str,
        context: str,
        answer: str,
        system_name: str = "AIJudge",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Evaluate an answer using LiteLLM. Returns a parsed JSON dict or
        an error-structure similar to the previous implementation.
        """
        logger.info(f"⚖️  Avaliando {system_name} com AIJudge (LiteLLM)...")

        if model is None:
            model = self._default_model()

        if model is None:
            raise ValueError("AIJudge: nenhum modelo especificado (model param or test_config.json)")

        if temperature is None:
            temperature = self._default_temperature()

        prompt = self.create_judge_prompt(question, context, answer)

        retry_delay = 5

        for attempt in range(max_retries):
            try:
                # Use response_format json_object to try get structured JSON from model
                response = completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    #max_tokens=1500,
                )

                # LiteLLM returns a Message object; cast to Any to bypass type checking
                response_any: Any = response
                response_text = (response_any.choices[0].message.content or "").strip()

                # Strip code fences if present
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]
                elif response_text.startswith('```'):
                    response_text = response_text[3:]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]

                try:
                    evaluation = json.loads(response_text)
                    logger.info(f"✅ Avaliação de {system_name} concluída (LiteLLM)")
                    return evaluation
                except json.JSONDecodeError:
                    # If parsing fails, include raw text in error for debugging
                    logger.error("AIJudge: não foi possível parsear JSON da resposta do modelo")
                    logger.debug(f"Resposta bruta: {response_text}")
                    return {"error": "invalid_json", "raw_response": response_text}

            except OpenAIError as oe:
                err = str(oe)
                logger.error(f"Erro OpenAIError: {err}")
                if '503' in err or 'overloaded' in err.lower():
                    if attempt < max_retries - 1:
                        logger.warning(f"Modelo sobrecarregado. Tentativa {attempt+1}/{max_retries}. Aguardando {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        logger.error("Falha por sobrecarga após tentativas")
                        break
                else:
                    break

            except Exception as e:
                err = str(e)
                logger.error(f"Erro ao avaliar: {err}")
                if '503' in err or 'overloaded' in err.lower():
                    if attempt < max_retries - 1:
                        logger.warning(f"Modelo possivelmente sobrecarregado. Tentativa {attempt+1}/{max_retries}. Aguardando {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                # For other exceptions, do not retry
                break

        # All retries failed
        return {
            "error": "Falha após todas as tentativas",
            "factual_consistency": {"score": 0, "justification": "Erro na avaliação"},
            "instruction_following": {"score": 0, "justification": "Erro na avaliação"},
            "domain_knowledge": {"score": 0, "justification": "Erro na avaliação"},
            "context_precision": {"score": 0, "justification": "Erro na avaliação"},
            "context_recall": {"score": 0, "justification": "Erro na avaliação"}
        }
