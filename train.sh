set -e
DATAPATH=../data/Multimodal_Retrieval
PRETRAIN_MODEL_PATH=../pretrain_models
LR=1e-5
BATCH_SIZE=36
PRECISION=amp
IMAG_MODEL=CLIP-zh-base
WARMUP=30
ADV_TRAIN=False
TEXT_SMOOTH=False
GROUP_CONTRASTIVE_LOSS_RATIO=0.0
TRAIN_SAMPLES=20000

OUTPUT_DIR=train_samples=${TRAIN_SAMPLES}_lr=${LR}_warmup=${WARMUP}_BatchSize=${BATCH_SIZE}_Precision=${PRECISION}_ImageModel=${IMAG_MODEL}_group-contrastive=${GROUP_CONTRASTIVE_LOSS_RATIO}_adv-train=${ADV_TRAIN}_text-smooth=${TEXT_SMOOTH}

export PYTHONPATH="$PYTHONPATH:$PWD/src"
export CUDA_VISIBLE_DEVICES=1,2,3,4,5



python -u src/training/main.py \
    --resume ${PRETRAIN_MODEL_PATH}/model-scope/clip-base-zh/pytorch_model.bin \
    --train_samples ${TRAIN_SAMPLES} \
    --save-frequency 1 \
    --group_contrastive_loss_ratio ${GROUP_CONTRASTIVE_LOSS_RATIO} \
    --group_contrastive_start_epoch 1 \
    --train-data="${DATAPATH}/MR_train_queries.jsonl"  \
    --train-img="${DATAPATH}/MR_train_imgs.224.npz"  \
    --warmup ${WARMUP} \
    --batch-size ${BATCH_SIZE} \
    --precision ${PRECISION} \
    --lr=${LR} \
    --wd=0.001 \
    --patience 6 \
    --epochs=2 \
    --name ${OUTPUT_DIR} \
    --model ${IMAG_MODEL}


export CUDA_VISIBLE_DEVICES=1

for CKPT_NAME in `ls logs/${OUTPUT_DIR}/checkpoints`;
do
    for mode in valid;
    do 
    python -u src/eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/MR_${mode}_imgs.224.npz" \
    --text-data="${DATAPATH}/MR_${mode}_queries.jsonl" \
    --image-feat-output-path "logs/${OUTPUT_DIR}/MR_${mode}_imgs.224.img_feat.${CKPT_NAME}.jsonl" \
    --text-feat-output-path "logs/${OUTPUT_DIR}/MR_${mode}_queries.txt_feat.${CKPT_NAME}.jsonl" \
    --img-batch-size=64 \
    --text-batch-size=128 \
    --resume="logs/${OUTPUT_DIR}/checkpoints/${CKPT_NAME}" \
    --model ${IMAG_MODEL}

    #! generate predictions
    python -u src/eval/make_topk_predictions.py \
    --image-feats="logs/${OUTPUT_DIR}/MR_${mode}_imgs.224.img_feat.${CKPT_NAME}.jsonl" \
    --text-feats="logs/${OUTPUT_DIR}/MR_${mode}_queries.txt_feat.${CKPT_NAME}.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="logs/${OUTPUT_DIR}/${mode}_predictions_${CKPT_NAME}.jsonl"


    #! eval score
    if [ ${mode} == valid ]; then
        python src/eval/evaluation.py ${DATAPATH}/MR_valid_queries.jsonl logs/${OUTPUT_DIR}/valid_predictions_${CKPT_NAME}.jsonl logs/${OUTPUT_DIR}/valid_${CKPT_NAME}_metrics.json
    fi
    done
done
