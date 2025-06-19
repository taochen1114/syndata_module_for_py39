import pandas as pd
import logging
import time
import os
from sdv.metadata import SingleTableMetadata

from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
"""
def get_data_from_model(model_path, num_rows=1000, condition_dict=None):
    print(f"Loading model from: {model_path}")

    if "GaussianCopula" in model_path:
        model = GaussianCopulaSynthesizer.load(model_path)
    elif "CTGAN" in model_path:
        model = CTGANSynthesizer.load(model_path)
    else:
        raise ValueError(f"Cannot determine model type from file name: {model_path}")

    if condition_dict:
        print(f"Sampling with conditions: {condition_dict}")
        return model.sample_conditions(conditions=condition_dict, num_rows=num_rows)
    else:
        return model.sample(num_rows=num_rows)
"""

def get_data_from_model(
    model_path,
    num_rows=1000,
    condition_dict=None,
    use_chunk=False,
    chunk_size=10,
    output_path=None
):
    print(f"Loading model from: {model_path}")

    # === 載入模型 ===
    if "GaussianCopula" in model_path:
        model = GaussianCopulaSynthesizer.load(model_path)
    elif "CTGAN" in model_path:
        model = CTGANSynthesizer.load(model_path)
    else:
        raise ValueError(f"Cannot determine model type from file name: {model_path}")

    # === 條件抽樣時，不支援 chunk ===
    if condition_dict:
        print(f"Sampling with conditions: {condition_dict}")
        return model.sample_conditions(conditions=condition_dict, num_rows=num_rows)

    # === Chunk 模式 ===    
    if use_chunk:
        assert output_path, "Chunk mode requires specifying output_path to save CSV."

        print(f"Sampling in chunks: total={num_rows}, chunk_size={chunk_size}")
        for i in range(0, num_rows, chunk_size):
            batch_size = min(chunk_size, num_rows - i)
            print(f" → Generating rows {i} ~ {i + batch_size}")

            chunk_df = model.sample(num_rows=batch_size)
            mode = 'w' if i == 0 else 'a'
            header = i == 0
            chunk_df.to_csv(output_path, index=False, mode=mode, header=header)

            del chunk_df  # 明確釋放記憶體

        print(f"✅ Finished sampling {num_rows} rows to: {output_path}")
        return None  # 若用 chunk 模式，不回傳整個 DataFrame

    # === 一次性抽樣 ===
    return model.sample(num_rows=num_rows)

def main(args):

    print("=== Generate Synthetic Data from Syn. Model ===")
    model_path = args.input_syn_model
    num_rows = args.num_rows
    output_fpath=args.output_fpath
    chunk_size = args.chunk_size

    print(f"The Syn. Model Path: {model_path}")
    print(f"Generate {num_rows} rows to {output_fpath}...")
    
    assert os.path.exists(model_path), f"Can't find the model_path pkl file: {model_path}."
    if args.use_chunk:
        print(f"=== use chunk mode size: {args.chunk_size} ===")

        get_data_from_model(
                model_path,
                num_rows = num_rows,
                use_chunk=True,
                chunk_size=chunk_size,
                output_path=output_fpath
            )
    else:
        output_df = get_data_from_model(model_path, num_rows=num_rows)
        print("output dataframe shape")
        print(output_df.shape)
        print("output dataframe head(5)")
        print(output_df.head())

        if args.save_output:
            print(f"=== Save csv to {output_fpath} ===")
            output_df.to_csv(output_fpath, index=False)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--synth_model", type=str, default="GaussianCopula", help="sythetic model type")
    parser.add_argument("--primary_key", type=str, default="", help="primary key in your tabular data")
    parser.add_argument("--save_model", action="store_true", help="set for save model pkl file")
    parser.add_argument("--save_output", action="store_true", help="set for save output csv file")
    parser.add_argument("--save_report", action="store_true", help="set for save report csv and image files")

    parser.add_argument("--input_syn_model", type=str, default=None, help="path to your syn_data model file")
    parser.add_argument("--num_rows", type=int, default=200, help="num rows of the output sythetic dataframe")
    parser.add_argument("--output_fpath", type=str, default=None, help="set full file path for your syn_data output csv")
    parser.add_argument("--chunk_size", type=int, default=0, help="chunk_size for sythetic data generator")
    parser.add_argument("--use_chunk", action="store_true", help="set for use chunk mode to generate synthetic data")

    args = parser.parse_args()
    
    start_time = time.time()
    main(args)
    time_cost = time.time() - start_time

    print(f"time_cost {time_cost}")

    real_data_df = pd.read_csv("input/data.csv")  # 真實資料路徑
    syn_data_df = pd.read_csv("output/syn_data.csv")  # 合成資料預設檔名為: 真實資料檔名 + "_GaussianCopula_output"

    print(real_data_df.head(3))
    print(syn_data_df.head(3))
