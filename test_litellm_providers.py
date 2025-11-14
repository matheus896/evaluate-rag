"""
Script de teste isolado para validar provedores LiteLLM.
Este arquivo é usado para depuração e validação de conectividade com cada provedor.
NÃO faz parte do fluxo principal de avaliação.

Uso:
    python test_litellm_providers.py
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Importar litellm
try:
    from litellm import completion
    import litellm
except ImportError:
    print("❌ ERRO: litellm não está instalado. Execute: pip install litellm>=1.79.3")
    exit(1)

# Carregar variáveis de ambiente
load_dotenv()

# Configuração de logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiteLLMProviderTester:
    """Classe para testar provedores LiteLLM de forma isolada."""

    def __init__(self):
        """Inicializa o testador."""
        self.results = {}
        self.provider_configs = {
            "provider": {
                "model": "cerebras/llama-3.3-70b",
                "api_key_env": "CEREBRAS_API_KEY",
                "capabilities": ["json_mode", "streaming", "low_cost"],
            },
        }

    def check_api_key(self, provider: str) -> bool:
        """Verifica se a chave de API está configurada."""
        api_key_env = self.provider_configs[provider].get("api_key_env")
        if api_key_env is None:
            logger.error(f"❌ {provider.upper()}: Chave de API não configurada no provider_configs")
            return False
        
        api_key = os.getenv(api_key_env)

        if not api_key:
            logger.error(f"❌ {provider.upper()}: Variável de ambiente '{api_key_env}' não configurada")
            return False

        if api_key.strip() == "":
            logger.error(f"❌ {provider.upper()}: Chave de API vazia")
            return False

        logger.info(f"✅ {provider.upper()}: Chave de API detectada")
        return True

    def test_provider_basic(self, provider: str) -> Optional[Dict[str, Any]]:
        """Testa connectividade básica com um provedor."""
        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 TESTE BÁSICO: {provider.upper()}")
        logger.info(f"{'='*80}\n")

        config = self.provider_configs.get(provider)
        if not config:
            logger.error(f"❌ Provedor '{provider}' não configurado")
            return None

        # Verificar chave de API
        if not self.check_api_key(provider):
            return None

        model = config["model"]
        logger.info(f"📦 Modelo: {model}")
        logger.info(f"✨ Capacidades: {', '.join(config['capabilities'])}\n")

        try:
            logger.info(f"🚀 Enviando teste básico para '{model}'...")
            start_time = time.time()

            response = completion(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "Responda com exatamente: OK",
                    }
                ],
                max_tokens=10,
                temperature=0.1,
            )

            elapsed_time = time.time() - start_time

            # Extrair resposta
            response_any: Any = response
            content = (response_any.choices[0].message.content or "").strip()
            logger.info(f"✅ SUCESSO em {elapsed_time:.2f}s")
            logger.info(f"📝 Resposta: '{content}'")

            return {
                "provider": provider,
                "model": model,
                "status": "success",
                "response": content,
                "latency": elapsed_time,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"❌ FALHA: {str(e)}")
            logger.exception("Exceção completa:")
            return {
                "provider": provider,
                "model": model,
                "status": "failed",
                "error": str(e),
                "timestamp": time.time(),
            }

    def test_provider_json_mode(self, provider: str) -> Optional[Dict[str, Any]]:
        """Testa modo JSON (necessário para AI Judge)."""
        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 TESTE JSON MODE: {provider.upper()}")
        logger.info(f"{'='*80}\n")

        config = self.provider_configs.get(provider)
        if not config:
            logger.error(f"❌ Provedor '{provider}' não configurado")
            return None

        if "json_mode" not in config.get("capabilities", []):
            logger.warning(f"⚠️ {provider.upper()} não suporta JSON mode")
            return None

        model = config["model"]
        logger.info(f"📦 Modelo: {model}")
        logger.info(f"🎯 Testando resposta JSON estruturada...\n")

        try:
            logger.info(f"🚀 Enviando teste JSON para '{model}'...")
            start_time = time.time()

            response = completion(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": 'Responda com um JSON contendo: {"teste": "ok", "status": "funcionando"}',
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=100,
                temperature=0.1,
            )

            elapsed_time = time.time() - start_time

            # Extrair e parsear JSON
            response_any: Any = response
            content = (response_any.choices[0].message.content or "").strip()
            try:
                json_data = json.loads(content)
                logger.info(f"✅ SUCESSO em {elapsed_time:.2f}s")
                logger.info(f"📊 JSON Parseado: {json.dumps(json_data, indent=2)}")

                return {
                    "provider": provider,
                    "model": model,
                    "status": "success",
                    "response": json_data,
                    "latency": elapsed_time,
                    "timestamp": time.time(),
                }

            except json.JSONDecodeError as je:
                logger.error(f"❌ Resposta não é JSON válido: {je}")
                logger.error(f"📝 Conteúdo recebido: {content}")
                return {
                    "provider": provider,
                    "model": model,
                    "status": "failed",
                    "error": f"JSON decode error: {str(je)}",
                    "raw_response": content,
                    "timestamp": time.time(),
                }

        except Exception as e:
            logger.error(f"❌ FALHA: {str(e)}")
            logger.exception("Exceção completa:")
            return {
                "provider": provider,
                "model": model,
                "status": "failed",
                "error": str(e),
                "timestamp": time.time(),
            }

    def test_provider_streaming(self, provider: str) -> Optional[Dict[str, Any]]:
        """Testa streaming (adicional, não crítico)."""
        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 TESTE STREAMING: {provider.upper()}")
        logger.info(f"{'='*80}\n")

        config = self.provider_configs.get(provider)
        if not config:
            logger.error(f"❌ Provedor '{provider}' não configurado")
            return None

        if "streaming" not in config.get("capabilities", []):
            logger.warning(f"⚠️ {provider.upper()} não suporta streaming")
            return None

        model = config["model"]
        logger.info(f"📦 Modelo: {model}")
        logger.info(f"🎯 Testando streaming...\n")

        try:
            logger.info(f"🚀 Iniciando stream para '{model}'...")
            start_time = time.time()

            response = completion(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "Conte até 3 rapidamente.",
                    }
                ],
                stream=True,
                max_tokens=50,
                temperature=0.1,
            )

            chunks_count = 0
            full_content = ""

            logger.info("📨 Recebendo chunks:\n")
            for chunk in response:
                chunk_any: Any = chunk
                if hasattr(chunk_any.choices[0].delta, 'content') and chunk_any.choices[0].delta.content:
                    content = chunk_any.choices[0].delta.content
                    full_content += content
                    chunks_count += 1
                    logger.info(f"  Chunk {chunks_count}: {content}")

            elapsed_time = time.time() - start_time
            logger.info(f"\n\n✅ SUCESSO em {elapsed_time:.2f}s")
            logger.info(f"📊 Total de chunks: {chunks_count}")
            logger.info(f"📝 Conteúdo completo: {full_content}")

            return {
                "provider": provider,
                "model": model,
                "status": "success",
                "chunks": chunks_count,
                "full_response": full_content,
                "latency": elapsed_time,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"❌ FALHA: {str(e)}")
            logger.exception("Exceção completa:")
            return {
                "provider": provider,
                "model": model,
                "status": "failed",
                "error": str(e),
                "timestamp": time.time(),
            }

    def run_all_tests(self, provider: str, run_streaming: bool = False):
        """Executa todos os testes para um provedor."""
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"# INICIANDO SUITE DE TESTES PARA: {provider.upper()}")
        logger.info(f"{'#'*80}\n")

        tests = [
            ("basic", self.test_provider_basic),
            ("json_mode", self.test_provider_json_mode),
        ]

        if run_streaming:
            tests.append(("streaming", self.test_provider_streaming))

        results = {}
        for test_name, test_func in tests:
            result = test_func(provider)
            results[test_name] = result

        # Resumo
        logger.info(f"\n\n{'='*80}")
        logger.info(f"📊 RESUMO DE TESTES: {provider.upper()}")
        logger.info(f"{'='*80}\n")

        passed = sum(1 for r in results.values() if r and r.get("status") == "success")
        failed = sum(1 for r in results.values() if r and r.get("status") == "failed")
        skipped = sum(1 for r in results.values() if r is None)

        logger.info(f"✅ Passaram: {passed}/{len(tests)}")
        logger.info(f"❌ Falharam: {failed}/{len(tests)}")
        logger.info(f"⊘ Pulados: {skipped}/{len(tests)}\n")

        for test_name, result in results.items():
            if result:
                status_icon = "✅" if result.get("status") == "success" else "❌"
                logger.info(f"{status_icon} {test_name.upper()}: {result.get('status', 'unknown')}")

        return results


def main():
    """Função principal."""
    print("\n" + "="*80)
    print("🧪 LITELLM PROVIDERS TEST SUITE")
    print("="*80)
    print("""
Este script valida a conectividade e funcionalidades de cada provedor LiteLLM
antes de integração com o sistema de avaliação RAG.

Testes executados:
  1. Teste Básico: Conectividade e resposta simples
  2. Teste JSON Mode: Modo JSON estruturado (crítico para AI Judge)
  3. Teste Streaming: Streaming de respostas (opcional)
""")
    print("="*80 + "\n")

    tester = LiteLLMProviderTester()

    # Testar Cerebras
    results = tester.run_all_tests("provider", run_streaming=False)

    # Relatório final
    print("\n" + "#"*80)
    print("# RELATÓRIO FINAL")
    print("#"*80 + "\n")

    all_success = all(
        r and r.get("status") == "success"
        for r in results.values()
    )

    if all_success:
        print("✅ TODOS OS TESTES PASSARAM!")
        print("\n🎉 O provedor está pronto para integração com o AI Judge.\n")
        return 0
    else:
        print("❌ ALGUNS TESTES FALHARAM!")
        print("\n⚠️ Verifique os erros acima e configure as variáveis de ambiente.\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
