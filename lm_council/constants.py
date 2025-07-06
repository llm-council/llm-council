# Pairwise choice constants.
MAJOR_A_WIN = "A>>B"
MINOR_A_WIN = "A>B"
MINOR_B_WIN = "B>A"
MAJOR_B_WIN = "B>>A"
TIE = "A=B"

FIVE_POINT_SCALE = [MAJOR_A_WIN, MINOR_A_WIN, TIE, MINOR_B_WIN, MAJOR_B_WIN]
FOUR_POINT_SCALE = [MAJOR_A_WIN, MINOR_A_WIN, MINOR_B_WIN, MAJOR_B_WIN]
THREE_POINT_SCALE = [MINOR_A_WIN, TIE, MINOR_B_WIN]
TWO_POINT_SCALE = [MINOR_A_WIN, MINOR_B_WIN]


# Mapping of LLM families to their colors for visualization purposes.
FAMILY_COLORS = {
    "openai": "mediumseagreen",
    "anthropic": "burlywood",
    "mistral": "darkorange",
    "google": "skyblue",
    "meta-llama": "magenta",
    "deepseek": "royalblue",
    "cohere": "darkslategray",
    "qwen": "slateblue",
    "council": "gold",
    "amazon": "orange",
    "x-ai": "black",
    "01-ai": "teal",
    "recursal": "darkslateblue",
}
