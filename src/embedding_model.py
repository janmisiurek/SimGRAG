from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI


class EmbeddingModel:
    def __init__(self, configs):
        self.configs = configs['embedding_model']
        provider = self.configs.get('provider', 'local')
        if provider == 'azure':
            self.client = AzureOpenAI(
                azure_endpoint=self.configs['endpoint'],
                api_key=self.configs['api_key'],
                api_version=self.configs['api_version']
            )
            self.deployment = self.configs['deployment']
            self.batch_size = self.configs.get('batch_size', 16)
            self.model = None
        else:
            device = self.configs['device']
            model_path = self.configs['model_path']
            self.model = SentenceTransformer(
                model_path, trust_remote_code=True, local_files_only=True, device=device)

    def encode(self, texts, batch_size=64, show_progress_bar=False):
        if self.configs.get('provider') == 'azure':
            if isinstance(texts, str):
                texts = [texts]
            vectors = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                resp = self.client.embeddings.create(
                    input=batch, model=self.deployment)
                vectors.extend([d.embedding for d in resp.data])
            return vectors
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar)

