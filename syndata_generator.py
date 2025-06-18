import pandas as pd
import logging
import time
import os
from sdv.metadata import SingleTableMetadata

from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer

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


def data_sythesizer(args, input_df=pd.DataFrame()):
    """Synthesize input dataframe data and return synthetic output."""

    # === Step 1: 若有 primary key，先轉為 str 避免 regex 檢查錯誤 ===
    pri_key = args.primary_key
    if pri_key and pri_key in input_df.columns:
        input_df[pri_key] = input_df[pri_key].astype(str)

    # === Step 2: 建構 metadata ===
    # metadata = Metadata.detect_from_dataframe(data=input_df)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=input_df)

    if pri_key:
        metadata.update_column(column_name=pri_key, sdtype="id")
        metadata.set_primary_key(pri_key)
        logging.info(f"Primary key '{pri_key}' set as sdtype='id'.")

    # === Step 3: Initialize model ===
    if args.synth_model == "GaussianCopula":
        print("Synthetic model arch: GaussianCopula")
        model = GaussianCopulaSynthesizer(metadata)
    elif args.synth_model == "CTGAN":
        if args.custom_setting:
            print("Synthetic model arch: CTGAN (custom)")
            model = CTGANSynthesizer(
                metadata=metadata,
                epochs=args.epochs,
                batch_size=args.batch_size,
                generator_dim=tuple(args.gen_dim),
                discriminator_dim=tuple(args.dis_dim),
                verbose=True,
            )
        else:
            logging.info("Synthetic model arch: CTGAN (default)")
            model = CTGANSynthesizer(metadata, verbose=True)
    else:
        raise ValueError(f"Unsupported synth_model: {args.synth_model}")

    # === Step 4: Fit model ===
    print("Fitting synthetic model ...")
    start_time = time.time()
    model.fit(input_df)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")

    # === Step 5: Sample data ===
    output_df = model.sample(num_rows=args.num_rows)

    # === Step 6: Save model if specified ===
    if args.save_model:
        print("=== save Syn. Model file ===")
        model_name = f"syn_model_{args.synth_model}"
        if args.custom_setting:
            model_name += "-c"
        model_name += ".pkl"

        output_dir = os.path.dirname(args.output_fpath) if args.output_fpath else args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, model_name)

        print(f"Saving model to {model_path}")
        model.save(model_path)

    # === Step 7: Save output CSV if specified ===
    if args.save_output and args.output_fpath:
        os.makedirs(os.path.dirname(args.output_fpath), exist_ok=True)
        print(f"Saving synthetic output to {args.output_fpath}")
        output_df.to_csv(args.output_fpath, index=False)

    return output_df

def set_args(args_list=None):
    """Main Function
    process input and do configs check
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/train.csv")
    parser.add_argument("--output_dir", type=str, default="data/output/")
    parser.add_argument("--synth_model", type=str, default="GaussianCopula", help="sythetic model type")
    parser.add_argument("--primary_key", type=str, default="", help="primary key in your tabular data")
    parser.add_argument("--num_rows", type=int, default=200, help="num rows of the output sythetic dataframe")
    parser.add_argument("--save_model", action="store_true", help="set for save model pkl file")
    parser.add_argument("--save_output", action="store_true", help="set for save output csv file")
    parser.add_argument("--save_report", action="store_true", help="set for save report csv and image files")
    
    parser.add_argument("--custom_setting", action="store_true", help="set for custom setting in CTGAN and TVAE Model")
    parser.add_argument("--epochs", type=int, default=300, help="set epochs for training CTGAN and TVAE Model")
    parser.add_argument("--batch_size", type=int, default=500, help="set batch size for training CTGAN and TVAE Model")
    parser.add_argument("--gen_dim", type=int, nargs="+", default=[256, 256], help="set gen dimension")
    parser.add_argument("--dis_dim", type=int, nargs="+", default=[256, 256], help="set dis dimension")
    
    parser.add_argument("--input_syn_model", type=str, default=None, help="path to your syn_data model file")
    parser.add_argument("--sample_condition", type=str, default=None, help="path to your syn_data sample condition json file")
    parser.add_argument("--output_fpath", type=str, default=None, help="set full file path for your syn_data output csv")

    return parser.parse_args(args_list)


def main(args):
    # logging.info(f"contents of args.primary_key {args.primary_key}")
    # logging.info(f"contents of args.custom_setting {args.custom_setting}")
    # logging.info(f"contents of args.gen_dim {args.gen_dim}")
    # logging.info(f"contents of args.dis_dim {args.dis_dim}")
    if not args.input_syn_model:
        print("=== Train Synthetic Model and Generate Sample SynData csv ===")
        assert os.path.exists(args.input_path), f"Can't find the input file at {args.input_path}."
        assert os.path.exists(args.output_dir), f"Can't find the output folder at {args.output_dir}."
        assert args.synth_model in ["GaussianCopula", "CTGAN"]
        # ["GaussianCopula", "CTGAN", "CopulaGAN", "TVAE"]

        
        input_path = args.input_path
        output_dir = args.output_dir
        # output_fname=args.output_fname

        print(f"input file path: {input_path}")
        print(f"output directory: {output_dir}")

        input_df = pd.read_csv(input_path)
        if args.primary_key:
            assert args.primary_key in input_df.columns

        # if "Id" in input_df.columns:
        #     input_df = input_df.drop(columns=["Id"])

        output_df = data_sythesizer(args=args, input_df=input_df)

        logging.info("output dataframe shape")
        logging.info(output_df.shape)
        logging.info("output dataframe head(5)")
        logging.info(output_df.head())


        if args.save_output:
            print("=== save output csv file ===")
            
            base = os.path.basename(input_path)
            output_fname = (
                os.path.splitext(base)[0] + "_" + args.synth_model + "_output.csv"
            )
            output_df.to_csv(os.path.join(output_dir, output_fname), index=False)
            
            print(f"saved to {os.path.join(output_dir, output_fname)}")

    else:
        print("=== Generate Synthetic Data from Syn. Model ===")
        model_path = args.input_syn_model
        num_rows = args.num_rows
        condition_fpath = args.sample_condition
        output_fpath=args.output_fpath

        print(f"The Syn. Model Path: {model_path}")
        print(f"Generate {num_rows} rows to {output_fpath}...")
        
        assert os.path.exists(model_path), f"Can't find the model_path pkl file: {model_path}."
        if not condition_fpath:
            output_df = get_data_from_model(model_path, num_rows=num_rows, condition_dict=None)
        else:
            assert os.path.exists(condition_fpath), f"Can't find the sample_condition json file: {condition_fpath}."
            assert condition_fpath[-5:] == ".json", f"{condition_fpath} must be a json file!"
            
            with open(condition_fpath, "r") as f:
                condition_dict = json.load(f)

            output_df = get_data_from_model(model_path, num_rows=num_rows, condition_dict=condition_dict)

        print("output dataframe shape")
        print(output_df.shape)
        print("output dataframe head(5)")
        print(output_df.head())

        if args.save_output:
            print(f"=== Save csv to {output_fpath} ===")
            output_df.to_csv(output_fpath, index=False)


if __name__ == "__main__":

    args = set_args([
        "--input_syn_model", "output/syn_model_GaussianCopula.pkl", # 合成資料生成模型路徑 
        "--output_fpath", "output/syn_data.csv",   # 合成資料輸出路徑
        "--num_rows", "10000000",  # 生成的資料筆數 一千萬筆
        "--save_output"
    ])
    
    start_time = time.time()
    main(args)
    time_cost = time.time() - start_time

    print(f"time_cost {time_cost}")

    real_data_df = pd.read_csv("input/data.csv")  # 真實資料路徑
    syn_data_df = pd.read_csv("output/syn_data.csv")  # 合成資料預設檔名為: 真實資料檔名 + "_GaussianCopula_output"

    print(real_data_df.head(3))
    print(syn_data_df.head(3))
