import sys
from inference import infer

prompts = sys.argv[1:]

infer(prompts[0])