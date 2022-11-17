export PYTHONPATH="$PYTHONPATH:$PWD/src"
DATAPATH=./logs/lr=8e-5_warmup=500_BatchSize=8_Precision=amp_ImageModel=ViT-B-16-2022-10-31:14:04:09
python -u src/eval/make_topk_predictions.py \
    --image-feats="${DATAPATH}/MR_test_imgs.224.img_feat.jsonl" \
    --text-feats="${DATAPATH}/MR_test_queries.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/test_predictions.jsonl"