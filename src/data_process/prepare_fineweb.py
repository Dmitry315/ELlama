import argparse
import logging
from datasets import load_dataset
import pandas as pd
logger = logging.getLogger(__name__)

def read_fine_web(save_path="../experiments/data/fineweb2"):
    logging.basicConfig(filename='prepare_fineweb_read_fine_web.log',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    dataset_name = "dmitry315/fineweb2-modern-greece-sample"
    logger.info('Start loading ' + dataset_name)
    ds = load_dataset(dataset_name)
    logger.info(dataset_name + " loaded. Start saving to jsonl: " + save_path)
    ds["train"].to_json(save_path + "/train.jsonl", lines=True, orient="records", force_ascii=False)
    ds["test"].to_json(save_path + "/val.jsonl", lines=True, orient="records", force_ascii=False)
    logger.info('Loaded to ' + save_path)

def save_txt(read_path="../experiments/data/fineweb2/train.jsonl", save_path="../experiments/data/fineweb2/train.txt"):
    logging.basicConfig(filename='prepare_fineweb_save_txt.log',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger.info('Read ' + read_path)
    df = pd.read_json(read_path, orient="records", lines=True)
    df["text"] = df["text"] + "\n\n"
    logger.info('Load to  ' + save_path)
    with open(save_path, "w") as f:
        f.writelines(df["text"].to_list())
    logger.info("Loaded to: " + save_path)
    

if __name__ == "__main__":
    logging.basicConfig(filename='prepare_fineweb.log',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    parser = argparse.ArgumentParser(description="Download Fineweb2 Greek")
    parser.add_argument("--save-path", type=str, default="../experiments/data/fineweb2/train.jsonl", help="Data save path")
    parser.add_argument("--to-txt",action="store_true",help="Convert to txt")
    parser.add_argument("--save-path-txt", type=str, default="../experiments/data/fineweb2/train.txt", help="Data save path")
    args = parser.parse_args()

    logger.info('Parameters')
    logger.info(str(args))
    
    read_fine_web(save_path=args.save_path)

    if args.to_txt:
        save_txt(read_path=args.save_path + "/train.jsonl", save_path=args.save_path_txt)
    logger.info('End Loading Fineweb')