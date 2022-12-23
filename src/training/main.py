import os
import time
import logging
from time import gmtime, strftime
from pathlib import Path
import json

import torch
from torch import optim

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler

from clip.clip import load
from clip.model import convert_weights, CLIP
from training.train import train, evaluate
from training.data import get_data
from training.params import parse_args
from training.logger import setup_primary_logging, setup_worker_logging
from training.scheduler import cosine_lr

# Used by https://github.com/openai/CLIP/issues/83 but not below.
# Keeping it incase needed.
def convert_models_to_fp32(model):
   
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def is_master(args):
    return (not args.distributed) or args.gpu == 0 or args.dp

def main_worker(gpu, ngpus_per_node, log_queue, args):
    args.gpu = gpu
    args.rank = gpu
    setup_worker_logging(args.rank, log_queue, args.log_level)

    # Log and save params.
    if is_master(args):
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")
            
    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    
    if args.dp:
        args.batch_size *= args.world_size

    if args.gpu is not None:
        logging.info(f"Use GPU: {args.gpu} for training")
        torch.cuda.set_device(args.gpu)
  
    model_config_file = Path(__file__).parent / f"model_configs/{args.model.replace('/', '-')}.json"
    print('Loading model from', model_config_file)
    assert os.path.exists(model_config_file)
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)
    
 
    model_info['text_smoothing'] = args.text_smoothing
    model = CLIP(**model_info)
 
    if args.clip_weight_path is not None:
        assert os.path.exists(args.clip_weight_path), "Pretrained CLIP weight not exists!"
    if args.bert_weight_path is not None:
        assert os.path.exists(args.bert_weight_path), "Pretrained BERT weight not exists!"
    load(model, clip_path=args.clip_weight_path, bert_path=args.bert_weight_path)

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32" or args.gpu is None:
        convert_models_to_fp32(model)

    if not torch.cuda.is_available():
        model.float()
        logging.warning("using CPU, this will be slow")
    else:
        model.cuda(args.gpu)
        if args.precision == "fp16":
            convert_weights(model)

        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.distributed:
         
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        if args.dp:
           
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)

        if args.precision == "fp16":
            convert_weights(model)

    data = get_data(args, max_txt_length=model_info['context_length'])

    exclude = lambda n : "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n : not exclude(n)

    named_parameters = list(model.named_parameters())
 
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
 
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    if args.train_data is None:
        optimizer = None
        scheduler = None
    else:
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.gpu is None:
          
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                
                checkpoint = torch.load(args.resume, map_location='cpu')
            # start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            
            model.to(loc)
            # if optimizer is not None:
            #     optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    cudnn.deterministic = False

    # determine if this worker should save logs and checkpoints.
    # only do so if it is the 0th worker.
    args.save_logs = (args.logs is not None and args.logs != '' and args.logs.lower() != 'none') and (
        (not args.distributed) or args.gpu == 0
    )


    best_val_loss = 100000

    flag_tensor = torch.zeros(1).to(args.gpu)
    
    early_stop_count = 0
    for epoch in range(start_epoch, args.epochs):
        if args.gpu == 0:
            logging.info(f'Start epoch {epoch}')
       
        train_loss = train(model, data, epoch, optimizer, scaler, scheduler, args)
  
        steps = data["train"].dataloader.num_batches * (epoch + 1)

        if args.val_data is not None:
            metrics = evaluate(model, data, epoch + 1, args, steps)

        # Saving checkpoints.
        
        logging.info(metrics)
        val_loss = metrics['val_loss']
       
        
        if args.save_logs and (args.gpu == 0 or (not args.distributed)):
            
            dist.all_reduce(val_loss)
            val_loss /= dist.get_world_size()
            val_loss = metrics['val_loss']
            if val_loss < best_val_loss:
                best_val_loss = vali_loss
                early_stop_count = 0
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(args.checkpoint_path, f"best_model.pt")
                    )
 
            logging.info(f'Current best train_loss:{best_val_loss}')
            

                
            
            #! save freq
            if args.save_frequency > 0 and ((epoch + 1) % args.save_frequency) == 0 and epoch + 1 != args.epochs // 2:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pt"),
                )
                
                
            if val_loss >= best_val_loss:
                early_stop_count += 1
                if early_stop_count == args.patience:
                    flag_tensor += 1

        dist.all_reduce(flag_tensor, op=dist.ReduceOp.SUM)
        if flag_tensor == 1:
            if args.gpu == 0:
                logging.info(f"=> {args.patience} Epochs No Improvements, Training early stop !!! ")
            break
                    
                    
            


def main():
    args = parse_args()

    # get the name of the experiments
    if args.name is None:
        args.name = strftime(
            f"lr={args.lr}_"
            f"agg={args.aggregate}_"
            f"model={args.model}_"
            f"batchsize={args.batch_size}_date=%Y-%m-%d-%H-%M-%S",
            gmtime(),
        )

    if args.copy_codebase:
        import sys, subprocess
        from shutil import copytree, ignore_patterns
        new_code_path = os.path.join(args.logs, args.name, "code")
        if os.path.exists(new_code_path) and os.path.listdir(new_code_path):
            print(
                f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
            )
            return -1
        if os.path.exists(new_code_path) and not os.path.listdir(new_code_path):
            print(
                f"Warning. Experiment already exists at {new_code_path}. But dir is empty"
            )
        print(f"Copying codebase to {new_code_path}")
        current_code_path = os.path.realpath(__file__)
        for _ in range(3):
            current_code_path = os.path.dirname(current_code_path)
        copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs'))
        print("Done copying code.")
        os.environ["PYTHONPATH"] = f"{os.environ['PYTHONPATH']}:{os.path.join(new_code_path, 'src')}"
        main_file = os.path.join(new_code_path, "src", "training", "main.py")
        argv = sys.argv
        argv.remove('--copy-codebase')
        argv.extend(['--name', args.name])
        command = [sys.executable] + argv
        print("Executing command:", " ".join(command))
        subprocess.check_call(command)
        return 1

    args.log_path = os.path.join(args.logs, args.name, "out.log")
    if os.path.exists(args.log_path):
        print(
            "Error. Experiment already exists. Use --name {} to specify a new experiment."
        )
        return -1

    assert args.precision in ['amp', 'fp16', 'fp32']

    args.ngpus_per_node = torch.cuda.device_count()

    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    for dirname in [args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)
    

    # Set multiprocessing type to spawn.
    # This is important for logging to work with multiprocessing.
    torch.multiprocessing.set_start_method("spawn")

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level)

    # Distributed training = training on more than one GPU.
    # Also easily possible to extend to multiple nodes & multiple GPUs.
    args.distributed = (args.gpu is None) and torch.cuda.is_available() and (not args.dp)
    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        args.world_size = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, log_queue, args))
    else:
        if args.dp:
            args.gpu = args.multigpu[0]
            args.world_size = len(args.multigpu)
        else:
            args.world_size = 1
        main_worker(args.gpu, None, log_queue, args)


if __name__ == "__main__":
    main()
