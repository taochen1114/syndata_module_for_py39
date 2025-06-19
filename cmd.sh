
echo "without chunking"

python syndata_generator.py \
    --input_syn_model output/syn_model_GaussianCopula.pkl \
    --output_fpath output/syn_data_1K.csv \
    --num_rows 1000 \
    --save_output



echo "with chunking"
python syndata_generator.py \
    --input_syn_model output/syn_model_GaussianCopula.pkl \
    --output_fpath output/syn_data_10M.csv \
    --num_rows 10000000 \
    --save_output \
    --use_chunk \
    --chunk_size 1000000

