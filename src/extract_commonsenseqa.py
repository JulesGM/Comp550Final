import json
import fire
import tqdm

def count_lines(path):
    with open(path) as fin:
        return sum(1 for _ in fin)

def main(input_path: str = "train_rand_split.jsonl",
         output_path: str = "flattened.txt"):
    total = count_lines(input_path)
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in tqdm.tqdm(fin, total=total):
            fout.write(json.loads(line)["question"]["stem"].strip() + "\n")

if __name__ == "__main__":
    fire.Fire(main)