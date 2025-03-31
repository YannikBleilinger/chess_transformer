from transformers import Trainer, TrainingArguments
from accelerate import Accelerator
from src.models.chessGPT_board_v2 import Transformer, Config
from src.models.ChessDataset import Chessset
from src.definition import BOARD_DATA
import os

'''
This training loop uses the hugginface 'trainer' api, for DP a accelerate config is neccessary
This training loop runs on two devices
'''

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
    LEARNING_RATE = 5e-5
    WARMUP_STEPS = 100
    WEIGHT_DECAY = 0.01
    BATCH_SIZE = 512

    accelerator = Accelerator()

    training_args = TrainingArguments(
        output_dir="./checkpoints/move_dist",
        num_train_epochs=2,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=2,
        do_eval=True,
        eval_strategy='steps',
        eval_steps=50_000,
        save_steps=20_000,
        logging_steps=100,
        logging_first_step=True,
        prediction_loss_only=False,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type='cosine',
        warmup_steps=WARMUP_STEPS,
        dataloader_num_workers=8,
        run_name="gpt_board",
        fp16=True,
        optim='adamw_torch_fused',
        torch_compile=True,
        no_cuda=False,
        ddp_find_unused_parameters=False
    )

    train_dataset = Chessset(BOARD_DATA, split='train')
    val_dataset = Chessset(BOARD_DATA, split='val')

    model = Transformer(Config())

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model parameters: {param_count}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.can_return_loss = True
    trainer.train()
