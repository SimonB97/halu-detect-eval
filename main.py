from detection.lbhd import LBHD
from models.llm import OpenAILlm

# Set Authentication Tokens in .env file!

# instantiate LLM models
llm = OpenAILlm("gpt-3.5-turbo")

# instantiate detection method objects
lbhd = LBHD(llm)

def run_evaluation():
    