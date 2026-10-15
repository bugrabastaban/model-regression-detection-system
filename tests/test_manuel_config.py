
from prompt_config import load_prompt_config

config = load_prompt_config()

print("Prompt yapılandırması:")
print("Versiyon:", config.version_id)
print("Zaman damgası:", config.timestamp)
print("System prompt:", config.system_prompt)
print("Few-shot örnek sayısı:", len(config.few_shot_examples))

assert config.version_id
assert config.system_prompt.strip()
assert len(config.few_shot_examples) > 0

print("\nPromptConfig testi başarılı!")
