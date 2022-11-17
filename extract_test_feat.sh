# only supports single-GPU inference
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PYTHONPATH:$PWD/src"
DATAPATH=../data/Multimodal_Retrieval
experiment_name=lr=8e-05_wd=0.001_agg=True_model=ViT-B-16_batchsize=32_date=2022-10-31-05-25-26

python -u src/eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/MR_test_imgs.224.npz" \
    --text-data="${DATAPATH}/MR_test_queries.jsonl" \
    --image-feat-output-path "logs/${experiment_name}/test_imgs.224.img_feat.jsonl" \
    --text-feat-output-path "logs/${experiment_name}/test_queries.txt_feat.jsonl" \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --resume="logs/${experiment_name}/checkpoints/epoch_5.pt" \
    --model ViT-B-16