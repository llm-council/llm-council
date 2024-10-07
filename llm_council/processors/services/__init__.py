from . import utils as utils
from . import base_service
from . import anthropic_service
from . import cohere_service
from . import lepton_service
from . import mistral_service
from . import openai_service
from . import together_service
from . import vertex_service

BaseService = base_service.BaseService
AnthropicService = anthropic_service.AnthropicService
CohereService = cohere_service.CohereService
LeptonService = lepton_service.LeptonService
MistralService = mistral_service.MistralService
OpenAIService = openai_service.OpenAIService
TogetherService = together_service.TogetherService
VertexService = vertex_service.VertexService

PROVIDER_REGISTRY = {
    "anthropic": AnthropicService,
    "mistral": MistralService,
    "cohere": CohereService,
    "lepton": LeptonService,
    "openai": OpenAIService,
    "together": TogetherService,
    "vertex": VertexService,
}
