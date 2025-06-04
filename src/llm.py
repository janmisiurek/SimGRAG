from openai import OpenAI, AzureOpenAI


class LLM:
    def __init__(self, configs):
        self.configs = configs['llm']
        print('Loading LLM ...')
        provider = self.configs.get('provider', 'openai')
        if provider == 'azure':
            self.client = AzureOpenAI(
                azure_endpoint=self.configs['endpoint'],
                api_key=self.configs['api_key'],
                api_version=self.configs['api_version']
            )
        else:
            self.client = OpenAI(
                base_url=self.configs.get('base_url'),
                api_key=self.configs['api_key']
            )
        print(self.chat("Hello!"))

    def chat(self, input_text):
        model_name = self.configs.get('deployment', self.configs.get('model'))
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": input_text}],
            temperature=self.configs.get('temperature', 0.0),
            top_p=self.configs.get('top_p', 1.0),
            max_tokens=self.configs.get('max_tokens', 1024)
        )
        return completion.choices[0].message.content.strip()

