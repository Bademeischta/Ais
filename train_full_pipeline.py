"""
Vollständiges Training: Phase 0 -> 1 -> 2 -> 3 + Evaluation
Nutzung: python train_full_pipeline.py
"""
import torch
import os
from cell4_training import Trainer, PPOTrainer, MODES, SAVE_DIR
from mechanics import HeadlessArena
from bots import RuleBasedBot

try:
    from evaluation import MetaLearningEvaluator
except ImportError:
    MetaLearningEvaluator = None


def main():
    print("=" * 60)
    print("  AIS META-LEARNING FULL TRAINING PIPELINE")
    print("=" * 60)

    trainer = Trainer()

    # Phase 0: Data Collection für CRN
    print("\n[Phase 0] Collecting CRN training data...")
    crn_dataset = trainer.collect_crn_data(num_episodes_per_mode=min(100, 30))

    # Phase 1: Supervised CRN Training (feature_encoder + CRN)
    print("\n[Phase 1] Training CRN (supervised)...")
    trainer.train_crn_supervised(
        crn=trainer.crn,
        encoder=trainer.feature_encoder,
        dataset=crn_dataset,
        epochs=50,
    )
    trainer.save_checkpoint("phase1_crn")

    # Phase 2: MSPN Training (isoliert pro Modus) – optional, verkürzt
    print("\n[Phase 2] Training MSPNs (isolated, 2 Episoden pro Modus als Demo)...")
    from cell4_training import DEVICE
    for mode_idx, mode_cls in enumerate(MODES):
        print(f"  Training MSPN {mode_idx} ({mode_cls.__name__})...")
        ppo_trainer = PPOTrainer(
            trainer.mspns[mode_idx], trainer.encoder, mode_idx, device=DEVICE
        )
        for ep in range(10):
            arena = HeadlessArena(mode_cls, [RuleBasedBot(i) for i in range(5)])
            ppo_trainer.collect_rollout(arena, steps=100)
            loss = ppo_trainer.update()
            if ep % 5 == 0:
                print(f"    Episode {ep}: Loss {loss:.4f}")
        trainer.save_checkpoint(f"phase2_mspn{mode_idx}")

    # Phase 3: MCN Meta-Training
    print("\n[Phase 3] Training MCN (meta-episodes)...")
    trainer.train_mcn_meta_episodes(num_episodes=50)
    trainer.save_checkpoint("phase3_mcn")

    # Evaluation
    if MetaLearningEvaluator is not None:
        print("\n[Evaluation] Running benchmarks...")
        evaluator = MetaLearningEvaluator(
            trainer.encoder,
            trainer.crn,
            trainer.mspns,
            trainer.mcn,
            MODES,
            device=DEVICE,
        )
        acc, conf = evaluator.evaluate_crn_accuracy(
            num_episodes_per_mode=10, feature_encoder=trainer.feature_encoder
        )
        transfer_results = evaluator.evaluate_transfer_quality(num_episodes=5)
        evaluator.plot_learning_curves(trainer.training_history)
        evaluator.generate_report()
    else:
        acc = trainer.evaluate_crn_accuracy(num_episodes_per_mode=10)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  CRN Accuracy: {acc*100:.2f}%")
    print(f"  Checkpoints saved in: {SAVE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
