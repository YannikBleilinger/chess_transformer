import math
from src.models.chessGPT_board_v2 import Transformer, Config
from src.models.ChessDataset import Chessset
from src.definition import BOARD_DATA

import time
import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

if __name__ == '__main__':
    num_workers = 8
    batch_size = 512
    accum_batch_size = batch_size * 2

    device = 'cuda:1'
    config = Config()
    model: Transformer = Transformer(config)
    model.to(device)
    model = torch.compile(model)

    train_dataset = Chessset(BOARD_DATA, split='train')
    val_dataset = Chessset(BOARD_DATA, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
    val_iter = iter(val_loader)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model parameters: {param_count}")
    torch.set_float32_matmul_precision("high")

    # gradient accumulation
    grad_accum_steps = int(accum_batch_size // batch_size)
    val_step = int((len(train_loader) // grad_accum_steps) / 3)
    print(
        f"Train loader: {train_loader.batch_size} batch size, len: {len(train_loader) // grad_accum_steps}, Accumulations steps: {grad_accum_steps}, Val interval: {val_step}")
    loss_accum = 0.0

    max_lr = 6e-4
    min_lr = max_lr * 0.1  # 10% of max
    warmup_steps = 100
    max_epochs = 3
    max_steps = (len(train_loader) / grad_accum_steps) * max_epochs
    max_steps -= max_steps * 0.05  # last 5% of training with min lr

    def get_lr_mult(it: int):
        if it < warmup_steps:
            return (1 / warmup_steps) * (it + 1)
        if it > max_steps:
            return 0.1
        # cos decay has the form of lr = min_lr + 0.5 (max_lr - min_lr) * (1 + cos((pi * current_step) / total_steps))
        decay_ration = (it - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ration))  # starts at one and goes to zero
        lr = min_lr + coeff * (max_lr - min_lr)
        return lr / max_lr  # lr relative to max_lr


    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)
    schueduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=get_lr_mult)
    scaler = torch.amp.grad_scaler.GradScaler(device=device)
    model.train()
    step = 0
    save_interval = int((len(train_loader) // grad_accum_steps) / 2)  # save 2 times per epoch
    val_interval = int(len(train_loader) // grad_accum_steps / 30)  # validate 30 times per epoch
    val_loss_steps = 20
    for epoch in range(max_epochs):
        micro_step = 0
        t0 = time.time()
        for batch in train_loader:
            x, y = batch["tokens"], batch['targets']
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16):
                res = model(x, targets=y)
            loss = res['loss']
            loss = loss / grad_accum_steps
            loss_accum += loss
            loss = scaler.scale(loss)  # for mixed precision training use scaler
            loss.backward()
            if not micro_step == grad_accum_steps - 1:
                micro_step += 1
                continue
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer=optimizer)
            scaler.update()
            # optimizer.step()
            schueduler.step()
            optimizer.zero_grad()
            lr = schueduler.get_last_lr()[0]
            torch.cuda.synchronize()
            t1 = time.time()

            # evaluation + storage
            if step % 400 == 0:
                model.eval()
                val_loss_accum = 0.0
                with torch.no_grad():
                    for _ in range(val_loss_steps):
                        try:
                            batch_val = next(val_iter)
                            x_val, y_val = batch_val['tokens'], batch_val['targets']
                        except StopIteration:
                            val_iter = iter(val_loader)
                            x_val, y_val = next(val_iter)
                        x_val, y_val = x_val.to(device), y_val.to(device)
                        with torch.autocast(device_type=device, dtype=torch.float16):
                            res_val = model(x_val, y_val)
                        loss = res_val['loss']
                        loss = loss / val_loss_steps
                        val_loss_accum += loss.detach()
                print(f"Val loss: {val_loss_accum}")
                model.train()

            if step % val_step == 0:
                save_path = f"/home_local/ybleilinger/chessGPT/checkpoints/encoding_gpt{step}.pth"
                print(f"val loss: {val_loss_accum}, saving to : {save_path}")
                checkpoint = {
                    'model': model.state_dict(),
                    'config': model.config,
                    'step': step,
                    'optim': optimizer.state_dict()
                }
                torch.save(checkpoint, save_path)

            dt = (t1 - t0)
            games_per_sec = (batch_size * grad_accum_steps) / dt
            print(
                f"{datetime.datetime.now()} step: {step}, loss: {loss_accum}, clip_norm: {norm}, time: {dt * 1000}, games/s: {games_per_sec}, lr: {lr}")
            micro_step = 0
            loss_accum = 0.0
            step += 1
            t0 = time.time()

    save_path = f"/home_local/ybleilinger/chessGPT/checkpoints/encoding_gtp_final.pth"
    print(f"val loss: {val_loss_accum}, saving to : {save_path}")
    checkpoint = {
        'model': model.state_dict(),
        'config': model.config,
        'step': step,
        'optim': optimizer.state_dict()
    }
    torch.save(checkpoint, save_path)