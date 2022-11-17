PRETRAIN_MODEL_PATH=../pretrain_models/ViT-B-16
python src/preprocess/transform_openai_pretrain_weights.py \
    --raw-ckpt-path ${PRETRAIN_MODEL_PATH}/ViT-B-16.pt \
    --new-ckpt-path ${PRETRAIN_MODEL_PATH}/ViT-B-16.state_dict.pt