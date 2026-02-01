# cell4_training.py (Optimized for RTX 5070)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
import numpy as np
import os
import time
import random
from collections import deque

from mechanics import HeadlessArena
from entities import BaseBot, MAP_SIZE
from networks import SharedEncoder, CRN, MSPN, MCN
from gamemodes import ClassicArena, TagMode, TeamDeathmatchMode, CaptureTheFlagMode, KingOfTheHillMode, \
                      BattleRoyaleMode, InfectionMode, ResourceCollectorMode, RacingMode, PuzzleCooperationMode
from bots import RuleBasedBot, MetaBot

MODES = [ClassicArena, TagMode, TeamDeathmatchMode, CaptureTheFlagMode, KingOfTheHillMode,
         BattleRoyaleMode, InfectionMode, ResourceCollectorMode, RacingMode, PuzzleCooperationMode]

# Configuration from ENV_CONFIG
try:
    from cell1_setup import ENV_CONFIG
except ImportError:
    ENV_CONFIG = {"BATCH_SIZE": 64, "NUM_WORKERS": 2, "HAS_GPU": False, "SAVE_DIR": "./saves/"}

BATCH_SIZE = ENV_CONFIG.get("BATCH_SIZE", 512)
NUM_WORKERS = ENV_CONFIG.get("NUM_WORKERS", 8)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def persistent_worker(worker_id, task_queue, result_queue):
    """Worker that stays alive and runs multiple episodes."""
    print(f"👷 Worker {worker_id} ready.")
    while True:
        task = task_queue.get()
        if task is None: break # Shutdown

        mode_cls, weights = task
        # Local model initialization with shared weights
        # (In a real setup we'd use shared_memory for efficiency)

        bots = [RuleBasedBot(i) for i in range(10)]
        arena = HeadlessArena(mode_cls, bots)

        episode_data = []
        for _ in range(100):
            results, victory = arena.step()
            episode_data.append(results)
            if victory: break

        result_queue.put(episode_data)

class Trainer:
    def __init__(self):
        self.encoder = SharedEncoder().to(DEVICE)
        self.crn = CRN().to(DEVICE)
        self.mspns = nn.ModuleList([MSPN() for _ in range(10)]).to(DEVICE)
        self.mcn = MCN().to(DEVICE)
        
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.mspns.parameters()), lr=3e-4)
        self.scaler = GradScaler() if DEVICE.type == 'cuda' else None

        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers = []

    def start_workers(self):
        for i in range(NUM_WORKERS):
            p = mp.Process(target=persistent_worker, args=(i, self.task_queue, self.result_queue))
            p.start()
            self.workers.append(p)

    def stop_workers(self):
        for _ in range(NUM_WORKERS):
            self.task_queue.put(None)
        for p in self.workers:
            p.join()

    def train_step(self, batch):
        """Perform one training step with Mixed Precision."""
        self.optimizer.zero_grad()
        
        if self.scaler:
            with autocast():
                # Actual forward/loss logic here...
                # For demo, we'll just simulate a loss
                loss = torch.tensor(0.1, requires_grad=True).to(DEVICE)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = torch.tensor(0.1, requires_grad=True).to(DEVICE)
            loss.backward()
            self.optimizer.step()
        
        return loss.item()

    def run_phase2(self, num_iterations=10):
        print(f"🚀 Starting Phase 2 Training (Mixed Precision, {NUM_WORKERS} workers)")
        self.start_workers()

        for i in range(num_iterations):
            # Dispatch tasks
            for _ in range(NUM_WORKERS):
                mode_cls = random.choice(MODES)
                self.task_queue.put((mode_cls, None))

            # Collect results
            for _ in range(NUM_WORKERS):
                data = self.result_queue.get()
                # Process data and perform train_step...
                loss = self.train_step(data)

            if i % 2 == 0:
                print(f"  Iteration {i}: Loss {loss:.4f}")

        self.stop_workers()
        print("✅ Training sequence finished.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    trainer = Trainer()
    trainer.run_phase2(5)
