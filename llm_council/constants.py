LLM_COUNCIL_MEMBERS = [
    # OpenAI.
    "openai://gpt-3.5-turbo-0125",
    "openai://gpt-4-turbo-2024-04-09",  # gpt-4-turbo
    "openai://gpt-4-0613",  # gpt-4
    "openai://gpt-4o-mini-2024-07-18",
    "openai://gpt-4o-2024-08-06",
    "openai://o1-preview-2024-09-12",
    "openai://o1-mini-2024-09-12",
    # Google/Vertex.
    "vertex://gemini-1.0-pro",
    "vertex://gemini-1.5-pro-001",
    "vertex://gemini-1.5-flash-001",
    # Anthropic.
    "anthropic://claude-3-haiku-20240307",
    "anthropic://claude-3-sonnet-20240229",
    "anthropic://claude-3-opus-20240229",
    "anthropic://claude-3-5-sonnet-20240620",
    # Mistral.
    "mistral://mistral-large-latest",  # our top-tier reasoning model for high-complexity tasks
    "mistral://open-mixtral-8x22b",  # Best open source model to date.
    "mistral://open-mixtral-8x7b",  # Our first sparse mixture-of-experts released
    # Meta.
    # "together://meta-llama/Llama-3-70b-chat-hf",
    # "together://meta-llama/Llama-3-8b-chat-hf",
    "together://meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "together://meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    # "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    # Alibaba.
    "together://Qwen/Qwen2-72B-Instruct",
    "together://Qwen/Qwen1.5-72B-Chat",
    # DBRX.
    "together://databricks/dbrx-instruct",
    # Cohere.
    "cohere://command-r-plus",
    "cohere://command-r",
]
