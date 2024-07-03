import json

from src.FENICE import FENICE

if __name__ == "__main__":
    with open("example.jsonl") as f:
        inputs = [json.loads(l.strip()) for l in f.readlines()]
    inputs = inputs * 10
    fenice = FENICE()
    print(fenice.score_batch(inputs))
