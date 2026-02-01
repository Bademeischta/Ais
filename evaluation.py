# evaluation.py – Meta-Learning Evaluation, Benchmarks, Reports
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
import time

from mechanics import HeadlessArena
from features import extract_mode_features, get_mode_id_from_class


class MetaLearningEvaluator:
    def __init__(self, encoder, crn, mspns, mcn, modes, device=None):
        self.encoder = encoder
        self.crn = crn
        self.mspns = mspns
        self.mcn = mcn
        self.modes = modes
        self.device = device or torch.device("cpu")
        self.metrics = defaultdict(list)

    def evaluate_crn_accuracy(self, num_episodes_per_mode=50, feature_encoder=None):
        """Messe CRN-Genauigkeit auf ungesehenen Episoden. Nutzt feature_encoder falls vorhanden."""
        from bots import RuleBasedBot

        enc = feature_encoder if feature_encoder is not None else self.encoder
        correct, total = 0, 0
        n_modes = len(self.modes)
        confusion = np.zeros((n_modes, n_modes))

        for mode_cls in self.modes:
            target_id = get_mode_id_from_class(mode_cls)
            for ep in range(num_episodes_per_mode):
                arena = HeadlessArena(mode_cls, [RuleBasedBot(i) for i in range(5)])
                feature_seq = []
                for _ in range(100):
                    arena.step()
                    feat = extract_mode_features(arena.world)
                    feature_seq.append(feat)
                feat_tensor = torch.FloatTensor(np.array(feature_seq)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    latent_seq = enc(feat_tensor)
                    mode_probs = self.crn(latent_seq)
                    predicted_id = torch.argmax(mode_probs, dim=1).item()
                confusion[target_id, predicted_id] += 1
                if predicted_id == target_id:
                    correct += 1
                total += 1

        accuracy = correct / total if total else 0
        self.metrics["crn_accuracy"].append(accuracy)

        print("\n📊 CRN Evaluation:")
        print(f"  Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
        print("\n  Confusion Matrix:")
        print(confusion.astype(int))
        return accuracy, confusion

    def evaluate_adaptation_speed(self, mode_cls, baseline_performance, num_episodes=200):
        """Messe wie schnell Meta-Learner im Vergleich zu from-scratch lernt."""
        from bots import MetaBot, RuleBasedBot

        meta_bot = MetaBot(0, self.encoder, self.crn, self.mspns, self.mcn)
        arena = HeadlessArena(mode_cls, [meta_bot])
        episode_rewards = []
        for ep in range(num_episodes):
            total_reward = 0
            for _ in range(min(500, 300)):
                arena.step()
                total_reward += meta_bot.last_reward
                meta_bot.last_reward = 0
            episode_rewards.append(total_reward)
            arena = HeadlessArena(mode_cls, [meta_bot])
        threshold_70 = baseline_performance * 0.7
        episodes_to_70 = next(
            (i for i, r in enumerate(episode_rewards) if r >= threshold_70), num_episodes
        )
        print("\n⚡ Adaptation Speed:")
        print(f"  Baseline: {baseline_performance:.1f}")
        print(f"  Episodes to 70%: {episodes_to_70}")
        print(f"  Final performance: {np.mean(episode_rewards[-10:]):.1f}")
        self.metrics["adaptation_episodes"].append(episodes_to_70)
        return episode_rewards, episodes_to_70

    def evaluate_transfer_quality(self, num_episodes=20):
        """Vergleiche Meta-Learner vs. RuleBased-Baseline auf allen Modi."""
        from bots import MetaBot, RuleBasedBot

        results = {}
        for mode_idx, mode_cls in enumerate(self.modes):
            meta_perf = self._test_single_mode(mode_cls, use_meta=True, num_episodes=num_episodes)
            baseline_perf = self._test_single_mode(
                mode_cls, use_meta=False, num_episodes=num_episodes
            )
            results[mode_cls.__name__] = {
                "meta": meta_perf,
                "mspn_only": baseline_perf,
                "ratio": meta_perf / max(1e-6, baseline_perf),
            }
        print("\n🔄 Transfer Quality:")
        for name, res in results.items():
            print(
                f"  {name}: Meta {res['meta']:.1f} | Baseline {res['mspn_only']:.1f} | Ratio {res['ratio']:.2f}"
            )
        return results

    def _test_single_mode(self, mode_cls, use_meta=True, mspn_idx=0, num_episodes=20):
        from bots import MetaBot, RuleBasedBot

        if use_meta:
            bot = MetaBot(0, self.encoder, self.crn, self.mspns, self.mcn)
        else:
            bot = RuleBasedBot(0)
        arena = HeadlessArena(
            mode_cls, [bot] + [RuleBasedBot(i + 1) for i in range(4)]
        )
        total_rewards = []
        for _ in range(num_episodes):
            ep_reward = 0
            for _ in range(min(500, 300)):
                arena.step()
                ep_reward += bot.last_reward
                bot.last_reward = 0
            total_rewards.append(ep_reward)
            arena = HeadlessArena(
                mode_cls, [bot] + [RuleBasedBot(i + 1) for i in range(4)]
            )
        return float(np.mean(total_rewards))

    def plot_learning_curves(self, training_history, save_path="./logs/learning_curves.png"):
        """Visualisiere Lernkurven für alle Komponenten."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        if "crn_accuracy" in training_history and training_history["crn_accuracy"]:
            axes[0, 0].plot(training_history["crn_accuracy"])
        axes[0, 0].set_title("CRN Recognition Accuracy")
        axes[0, 0].set_xlabel("Training Iteration")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].grid(True)

        for mode_idx in range(10):
            key = f"mspn_{mode_idx}_reward"
            if key in training_history and training_history[key]:
                axes[0, 1].plot(training_history[key], label=f"Mode {mode_idx}", alpha=0.7)
        axes[0, 1].set_title("MSPN Training Rewards")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Avg Reward")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        if "mcn_selections" in training_history:
            data = training_history["mcn_selections"]
            axes[1, 0].bar(range(len(data)), data)
            axes[1, 0].set_title("MCN MSPN Selection Frequency")
            axes[1, 0].set_xlabel("MSPN Index")
            axes[1, 0].set_ylabel("Selection Count")

        if "transfer_ratios" in training_history and training_history["transfer_ratios"]:
            axes[1, 1].plot(training_history["transfer_ratios"])
            axes[1, 1].axhline(y=1.0, color="r", linestyle="--", label="Baseline")
        axes[1, 1].set_title("Transfer Quality (Meta/Baseline Ratio)")
        axes[1, 1].set_xlabel("Evaluation Checkpoint")
        axes[1, 1].set_ylabel("Performance Ratio")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\n📈 Learning curves saved to {save_path}")

    def generate_report(self, save_path="./logs/evaluation_report.json"):
        """Generiere umfassenden Evaluationsbericht."""
        adaptation_list = [m for m in self.metrics.get("adaptation_episodes", []) if m > 0]
        report = {
            "timestamp": time.time(),
            "metrics": dict(self.metrics),
            "summary": {
                "crn_best_accuracy": max(self.metrics.get("crn_accuracy", [0])),
                "avg_adaptation_speed": float(np.mean(adaptation_list)) if adaptation_list else 0,
            },
        }
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to {save_path}")
        return report
