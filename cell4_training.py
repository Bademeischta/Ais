# cell4_training.py – Curriculum Learning, Checkpoints, Evaluation
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
from features import extract_mode_features, get_mode_id_from_class

MODES = [ClassicArena, TagMode, TeamDeathmatchMode, CaptureTheFlagMode, KingOfTheHillMode,
         BattleRoyaleMode, InfectionMode, ResourceCollectorMode, RacingMode, PuzzleCooperationMode]

try:
    from cell1_setup import ENV_CONFIG
except ImportError:
    ENV_CONFIG = {"BATCH_SIZE": 64, "NUM_WORKERS": 2, "HAS_GPU": False, "SAVE_DIR": "./saves/"}

BATCH_SIZE = ENV_CONFIG.get("BATCH_SIZE", 512)
NUM_WORKERS = ENV_CONFIG.get("NUM_WORKERS", 8)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = ENV_CONFIG.get("SAVE_DIR", "./saves/")
LOG_DIR = ENV_CONFIG.get("LOG_DIR", "./logs/")

# Curriculum: Phase 1 = isoliert (100 Episoden pro Modus), Phase 2 = gemischt, Phase 3 = schneller Wechsel
CURRICULUM_PHASE1_EPISODES_PER_MODE = 100
CURRICULUM_PHASE2_EPISODES = 500
CURRICULUM_PHASE3_SWITCH_EVERY_FRAMES = 200

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

    def save_checkpoint(self, phase_name="phase2", step=0):
        """Hierarchisches Speichern: Encoder, CRN, MSPNs, MCN."""
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(SAVE_DIR, f"shared_encoder_{phase_name}.pth"))
        torch.save(self.crn.state_dict(), os.path.join(SAVE_DIR, f"crn_model_{phase_name}.pth"))
        torch.save(self.mcn.state_dict(), os.path.join(SAVE_DIR, f"mcn_model_{phase_name}.pth"))
        for i, mspn in enumerate(self.mspns):
            torch.save(mspn.state_dict(), os.path.join(SAVE_DIR, f"mspn_{i}_{phase_name}.pth"))
        print(f"  Checkpoint saved: {phase_name} step {step}")

    def load_checkpoint(self, phase_name="phase2"):
        """Lade Checkpoints für Resume."""
        enc_path = os.path.join(SAVE_DIR, f"shared_encoder_{phase_name}.pth")
        if os.path.exists(enc_path):
            self.encoder.load_state_dict(torch.load(enc_path, map_location=DEVICE))
            self.crn.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"crn_model_{phase_name}.pth"), map_location=DEVICE))
            self.mcn.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"mcn_model_{phase_name}.pth"), map_location=DEVICE))
            for i in range(len(self.mspns)):
                p = os.path.join(SAVE_DIR, f"mspn_{i}_{phase_name}.pth")
                if os.path.exists(p):
                    self.mspns[i].load_state_dict(torch.load(p, map_location=DEVICE))
            print(f"  Checkpoint loaded: {phase_name}")

    def run_phase1_isolated(self, episodes_per_mode=100):
        """Phase 1: Jeden Modus isoliert trainieren (CRN-Labels = Modus-ID)."""
        print("📚 Phase 1: Isolated mode training")
        for mode_idx, mode_cls in enumerate(MODES):
            for ep in range(episodes_per_mode):
                arena = HeadlessArena(mode_cls, [RuleBasedBot(i) for i in range(10)])
                for _ in range(500):
                    results, victory = arena.step()
                    if victory: break
            if (mode_idx + 1) % 2 == 0:
                self.save_checkpoint("phase1", mode_idx)
        print("✅ Phase 1 done.")

    def run_phase2_mixed(self, num_iterations=100):
        """Phase 2: Zufälliger Modus pro Episode (gemischt)."""
        print(f"🚀 Phase 2: Mixed modes ({NUM_WORKERS} workers)")
        self.start_workers()
        for i in range(num_iterations):
            for _ in range(NUM_WORKERS):
                mode_cls = random.choice(MODES)
                self.task_queue.put((mode_cls, None))
            for _ in range(NUM_WORKERS):
                data = self.result_queue.get()
                loss = self.train_step(data)
            if i % 10 == 0:
                self.save_checkpoint("phase2", i)
                print(f"  Iteration {i}: Loss {loss:.4f}")
        self.stop_workers()
        print("✅ Phase 2 done.")

    def run_phase3_rapid_switch(self, num_meta_episodes=50):
        """Phase 3: Modus wechselt alle N Frames (Rapid Switch)."""
        print("⚡ Phase 3: Rapid mode switching")
        for _ in range(num_meta_episodes):
            mode_cls = random.choice(MODES)
            arena = HeadlessArena(mode_cls, [RuleBasedBot(i) for i in range(10)])
            for step in range(1000):
                results, victory = arena.step()
                if step > 0 and step % CURRICULUM_PHASE3_SWITCH_EVERY_FRAMES == 0:
                    mode_cls = random.choice(MODES)
                    arena.world.set_mode(mode_cls)
                if victory: break
        self.save_checkpoint("phase3", 0)
        print("✅ Phase 3 done.")

    def evaluate_crn_accuracy(self, num_episodes_per_mode=20):
        """Evaluation: CRN-Genauigkeit auf gelabelten Episoden (Stub)."""
        os.makedirs(LOG_DIR, exist_ok=True)
        correct, total = 0, 0
        for mode_cls in MODES:
            target_id = get_mode_id_from_class(mode_cls)
            for _ in range(num_episodes_per_mode):
                arena = HeadlessArena(mode_cls, [RuleBasedBot(i) for i in range(5)])
                feats = []
                for _ in range(100):
                    arena.step()
                    feats.append(extract_mode_features(arena.world))
                total += 1
                # Hier: CRN mit feat-Sequenz füttern, Vorhersage mit target_id vergleichen
                # correct += 1 if pred == target_id else 0
        acc = correct / total if total else 0
        print(f"  CRN Accuracy (stub): {acc*100:.1f}%")
        return acc


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    trainer = Trainer()
    trainer.run_phase1_isolated(episodes_per_mode=5)  # Kurz für Test
    trainer.run_phase2_mixed(num_iterations=10)
    trainer.run_phase3_rapid_switch(num_meta_episodes=3)
    trainer.evaluate_crn_accuracy(num_episodes_per_mode=2)
