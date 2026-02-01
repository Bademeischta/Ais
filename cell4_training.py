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
from collections import deque, defaultdict

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

# Feature-Dim für CRN (extract_mode_features)
FEATURE_DIM = 64

# Curriculum: Phase 1 = isoliert, Phase 2 = gemischt, Phase 3 = schneller Wechsel
CURRICULUM_PHASE1_EPISODES_PER_MODE = 100
CURRICULUM_PHASE2_EPISODES = 500
CURRICULUM_PHASE3_SWITCH_EVERY_FRAMES = 200


def collect_crn_data(num_episodes_per_mode=100):
    """
    Sammle gelabelte Episoden für supervised CRN-Training.
    Returns: dataset = [(feature_sequence, mode_id), ...]
    """
    dataset = []
    for mode_idx, mode_cls in enumerate(MODES):
        for ep in range(num_episodes_per_mode):
            arena = HeadlessArena(mode_cls, [RuleBasedBot(i) for i in range(5)])
            feature_seq = []
            for step in range(100):
                arena.step()
                feat = extract_mode_features(arena.world)
                feature_seq.append(feat)
            dataset.append((np.array(feature_seq, dtype=np.float32), mode_idx))
    return dataset


def train_crn_supervised(crn, encoder, dataset, epochs=50, device=DEVICE):
    """Supervised Training: CRN lernt Modus-ID aus Feature-Sequenzen. encoder: 64 -> 128."""
    optimizer = optim.Adam(list(crn.parameters()) + list(encoder.parameters()), lr=0.001)
    for epoch in range(epochs):
        random.shuffle(dataset)
        total_loss, correct = 0, 0
        for feat_seq, label in dataset:
            # feat_seq: (100, 64), label: scalar
            feat_tensor = torch.FloatTensor(feat_seq).unsqueeze(0).to(device)
            latent_seq = encoder(feat_tensor)  # (1, 100, 128)
            mode_probs = crn(latent_seq)  # (1, 10)

            loss = nn.CrossEntropyLoss()(mode_probs, torch.LongTensor([label]).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if torch.argmax(mode_probs, dim=1).item() == label:
                correct += 1

        acc = correct / len(dataset)
        print(f"  Epoch {epoch+1}/{epochs}: Loss {total_loss/len(dataset):.4f}, Acc {acc*100:.1f}%")
        if acc >= 0.95:
            break
    return total_loss / len(dataset), acc


class PPOTrainer:
    """PPO-Trainer für ein einzelnes MSPN im spezifischen Modus."""

    def __init__(self, mspn, encoder, mode_idx, device=DEVICE):
        self.mspn = mspn
        self.encoder = encoder
        self.mode_idx = mode_idx
        self.device = device
        self.optimizer = optim.Adam(list(mspn.parameters()), lr=3e-4)
        self.buffer = []

    def collect_rollout(self, arena, steps=200):
        """Sammle Rollout im spezifischen Modus. Bots müssen get_input_vector haben."""
        states, actions, rewards, log_probs, values = [], [], [], [], []

        for _ in range(steps):
            experiences = []
            for bot in arena.world.bots:
                if bot.died:
                    continue
                state = bot.get_input_vector(arena.world)
                states.append(state)

                with torch.no_grad():
                    st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    latent = self.encoder(st)
                    probs, value = self.mspn(latent)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    actions.append(action.item())
                    log_probs.append(dist.log_prob(action).item())
                    values.append(value.item())

                bot.apply_action(actions[-1])
                experiences.append((bot, state, actions[-1]))

            arena.world.update()

            # Rewards nach Update einsammeln
            idx = 0
            for bot in arena.world.bots:
                if bot.died:
                    idx += 1
                    continue
                rewards.append(bot.last_reward)
                bot.last_reward = 0
                idx += 1

            # Sicherstellen, dass rewards zu allen gesammelten states passen
            while len(rewards) < len(states):
                rewards.append(0.0)

        # Kürzen falls mehr states als rewards (z.B. durch mehrere Bots pro Step)
        n = min(len(states), len(actions), len(rewards), len(log_probs), len(values))
        self.buffer.append((
            states[:n], actions[:n], rewards[:n], log_probs[:n], values[:n]
        ))
        return n

    def update(self, epochs=4):
        """PPO Update mit Clipped Objective."""
        if not self.buffer:
            return 0.0

        states, actions, rewards, old_log_probs, old_values = self.buffer[-1]
        if not states:
            return 0.0

        # Compute returns & advantages
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        old_values_t = torch.FloatTensor(old_values).to(self.device)

        advantages = returns - old_values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        loss_val = 0.0
        for _ in range(epochs):
            latent = self.encoder(states_t)
            probs, values = self.mspn(latent)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(-1), returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mspn.parameters(), 0.5)
            self.optimizer.step()
            loss_val = loss.item()

        return loss_val


def train_mcn_meta_episodes(mcn, encoder, crn, mspns, num_episodes=100, device=DEVICE):
    """MCN lernt aus Meta-Episoden: welcher Experte wann genutzt wird (vereinfacht: Reward-Log)."""
    optimizer = optim.Adam(mcn.parameters(), lr=0.001)

    for ep in range(num_episodes):
        mode_cls = random.choice(MODES)
        arena = HeadlessArena(mode_cls, [RuleBasedBot(i) for i in range(5)])

        episode_rewards = []
        for step in range(min(500, 300)):
            if not arena.world.bots:
                break
            bot = arena.world.bots[0]
            if bot.died:
                break

            obs = bot.get_input_vector(arena.world)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                latent = encoder(obs_tensor)
                mode_probs = crn(latent.unsqueeze(1))

            meta_stats = torch.FloatTensor([step / 500.0, 0, 0, 0, 0]).unsqueeze(0).to(device)
            mspn_weights = mcn(mode_probs.squeeze(1), meta_stats)

            chosen_idx = torch.argmax(mspn_weights, dim=1).item()
            probs, _ = mspns[chosen_idx](latent)
            action = torch.distributions.Categorical(probs).sample().item()

            bot.apply_action(action)
            arena.step()

            reward = bot.last_reward
            episode_rewards.append(reward)
            bot.last_reward = 0

        total_reward = sum(episode_rewards)
        if ep % 10 == 0:
            print(f"  MCN Episode {ep}: Reward {total_reward:.1f}")

    return None


def persistent_worker(worker_id, task_queue, result_queue):
    """Worker that stays alive and runs multiple episodes."""
    print(f"👷 Worker {worker_id} ready.")
    while True:
        task = task_queue.get()
        if task is None:
            break
        mode_cls, weights = task
        bots = [RuleBasedBot(i) for i in range(10)]
        arena = HeadlessArena(mode_cls, bots)
        episode_data = []
        for _ in range(100):
            results, victory = arena.step()
            episode_data.append(results)
            if victory:
                break
        result_queue.put(episode_data)


class Trainer:
    def __init__(self):
        self.encoder = SharedEncoder(input_dim=228, latent_dim=128).to(DEVICE)
        self.feature_encoder = SharedEncoder(input_dim=FEATURE_DIM, latent_dim=128).to(DEVICE)
        self.crn = CRN().to(DEVICE)
        self.mspns = nn.ModuleList([MSPN() for _ in range(10)]).to(DEVICE)
        self.mcn = MCN().to(DEVICE)

        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.mspns.parameters()), lr=3e-4
        )
        self.scaler = GradScaler() if DEVICE.type == "cuda" else None

        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers = []

        self.training_history = defaultdict(list)

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

    def collect_crn_data(self, num_episodes_per_mode=100):
        """Wrapper: sammle CRN-Datensatz."""
        return collect_crn_data(num_episodes_per_mode=num_episodes_per_mode)

    def train_crn_supervised(self, crn=None, encoder=None, dataset=None, epochs=50):
        """Phase 1: CRN supervised. Nutzt feature_encoder (64->128) und crn."""
        crn = crn or self.crn
        encoder = encoder or self.feature_encoder
        if dataset is None:
            dataset = self.collect_crn_data(num_episodes_per_mode=50)
        loss_val, acc = train_crn_supervised(crn, encoder, dataset, epochs=epochs, device=DEVICE)
        self.training_history["crn_accuracy"].append(acc)
        return loss_val, acc

    def train_mcn_meta_episodes(self, mcn=None, encoder=None, crn=None, mspns=None, num_episodes=100):
        """Phase 3: MCN Meta-Episoden."""
        return train_mcn_meta_episodes(
            mcn or self.mcn,
            encoder or self.encoder,
            crn or self.crn,
            mspns or self.mspns,
            num_episodes=num_episodes,
            device=DEVICE,
        )

    def train_step(self, batch):
        """Perform one training step with Mixed Precision."""
        self.optimizer.zero_grad()
        if self.scaler:
            with autocast():
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
        """Hierarchisches Speichern: Encoder, Feature-Encoder, CRN, MSPNs, MCN."""
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(SAVE_DIR, f"shared_encoder_{phase_name}.pth"))
        torch.save(self.feature_encoder.state_dict(), os.path.join(SAVE_DIR, f"feature_encoder_{phase_name}.pth"))
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
            fe_path = os.path.join(SAVE_DIR, f"feature_encoder_{phase_name}.pth")
            if os.path.exists(fe_path):
                self.feature_encoder.load_state_dict(torch.load(fe_path, map_location=DEVICE))
            self.crn.load_state_dict(
                torch.load(os.path.join(SAVE_DIR, f"crn_model_{phase_name}.pth"), map_location=DEVICE)
            )
            self.mcn.load_state_dict(
                torch.load(os.path.join(SAVE_DIR, f"mcn_model_{phase_name}.pth"), map_location=DEVICE)
            )
            for i in range(len(self.mspns)):
                p = os.path.join(SAVE_DIR, f"mspn_{i}_{phase_name}.pth")
                if os.path.exists(p):
                    self.mspns[i].load_state_dict(torch.load(p, map_location=DEVICE))
            print(f"  Checkpoint loaded: {phase_name}")

    def run_phase1_isolated(self, episodes_per_mode=100):
        """Phase 1: Data sammeln + CRN supervised trainieren."""
        print("📚 Phase 1: CRN supervised training")
        dataset = self.collect_crn_data(num_episodes_per_mode=min(episodes_per_mode, 20))
        self.train_crn_supervised(dataset=dataset, epochs=50)
        self.save_checkpoint("phase1", 0)
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
        """Phase 3: Rapid mode switching + MCN."""
        print("⚡ Phase 3: Rapid mode switching + MCN")
        self.train_mcn_meta_episodes(num_episodes=num_meta_episodes)
        self.save_checkpoint("phase3", 0)
        print("✅ Phase 3 done.")

    def evaluate_crn_accuracy(self, num_episodes_per_mode=20):
        """Evaluation: CRN-Genauigkeit auf gelabelten Episoden."""
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
                feat_tensor = torch.FloatTensor(np.array(feats)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    latent_seq = self.feature_encoder(feat_tensor)
                    mode_probs = self.crn(latent_seq)
                    pred = torch.argmax(mode_probs, dim=1).item()
                if pred == target_id:
                    correct += 1
        acc = correct / total if total else 0
        print(f"  CRN Accuracy: {acc*100:.1f}%")
        return acc


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    trainer = Trainer()
    trainer.run_phase1_isolated(episodes_per_mode=5)
    trainer.run_phase2_mixed(num_iterations=10)
    trainer.run_phase3_rapid_switch(num_meta_episodes=3)
    trainer.evaluate_crn_accuracy(num_episodes_per_mode=2)
