import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from collections import defaultdict

# Import from backend
try:
    from app import (HeadlessArena, DQNBot, PPOBot, SniperBot, SurvivalBot, 
                    RandomBot, RuleBasedBot, GeneticBot, ActorCriticBot, 
                    SAVE_DIR, device)
except ImportError:
    # This might happen in the sandbox; usually app.py is created by cell3
    pass

# --- Configuration ---
NUM_ARENAS = 20
EPISODES = 100
TRAIN_SAVE_DIR = '/content/drive/MyDrive/Ais_Training/'
os.makedirs(TRAIN_SAVE_DIR, exist_ok=True)

def plot_training_results(stats):
    """
    Plots the training results as requested: Learning Curve, Killer Stats, and Bot Comparison.
    """
    plt.figure(figsize=(20, 10))
    
    # 1. Lernkurve (Average Reward over Episodes)
    plt.subplot(2, 2, 1)
    for algo, rewards in stats['rewards'].items():
        if len(rewards) > 0:
            smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
            plt.plot(smoothed, label=algo)
    plt.title("Lernkurve: Durchschnittliche Belohnung")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    
    # 2. Killer-Statistik (Who killed whom how often?)
    plt.subplot(2, 2, 2)
    algos = list(stats['kills'].keys())
    kills = [np.mean(stats['kills'][a]) for a in algos]
    plt.bar(algos, kills, color='salmon')
    plt.title("Killer-Statistik: Durchschnittliche Kills")
    plt.xticks(rotation=45)
    plt.ylabel("Kills")
    
    # 3. Win Rate & Accuracy (New)
    plt.subplot(2, 2, 3)
    win_rates = [np.mean(stats['wins'][a]) * 100 for a in algos]
    accuracies = [np.mean(stats['accuracy'][a]) * 100 for a in algos]
    x = np.arange(len(algos))
    plt.bar(x - 0.2, win_rates, 0.4, label='Win Rate %')
    plt.bar(x + 0.2, accuracies, 0.4, label='Genauigkeit %')
    plt.xticks(x, algos, rotation=45)
    plt.title("Erfolgsmetriken: Win Rate & Genauigkeit")
    plt.legend()
    
    # 4. Bot-Vergleich (Aggressivität vs Überleben)
    plt.subplot(2, 2, 4)
    for algo in algos:
        avg_survival = np.mean(stats['survival'][algo])
        avg_aggression = np.mean(stats['kills'][algo])
        plt.scatter(avg_survival, avg_aggression, label=algo, s=150)
        plt.text(avg_survival+5, avg_aggression, algo)
    plt.title("Bot-Vergleich: Aggressivität vs. Überleben")
    plt.xlabel("Durchschn. Überlebensdauer (Frames)")
    plt.ylabel("Durchschn. Kills")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(TRAIN_SAVE_DIR, "training_results.png"))
    plt.show()

def train():
    print(f"🚀 Initializing Headless Parallel Training on {device}...")
    
    stats = {
        'rewards': defaultdict(list),
        'kills': defaultdict(list),
        'deaths': defaultdict(list),
        'survival': defaultdict(list),
        'max_mass': defaultdict(list),
        'wins': defaultdict(list),
        'accuracy': defaultdict(list)
    }
    
    bot_types = [RandomBot, RuleBasedBot, GeneticBot, DQNBot, PPOBot, SniperBot, SurvivalBot]
    arenas = []
    
    # Create Arenas
    for i in range(NUM_ARENAS):
        arena_bots = []
        for j, b_type in enumerate(bot_types):
            arena_bots.append(b_type(i * 100 + j)) # 1 instance of each type per arena
        arenas.append(HeadlessArena(arena_bots))
    
    start_time = time.time()
    
    for ep in range(EPISODES):
        # Reset Stats for episode
        ep_winners = []
        
        # Simulation Loop (1000 frames)
        for frame in range(1000):
            # 1. Sense (Parallelized logically)
            for arena in arenas:
                for b in arena.bots:
                    b.sense(arena)
            
            # 2. Decide & Step (Physics)
            for arena in arenas:
                for b in arena.bots:
                    inp = b.get_input_vector()
                    action = b.decide(inp)
                    b.apply_action(action)
                arena.step()
                
            # 3. Learn
            for arena in arenas:
                for b in arena.bots:
                    if hasattr(b, 'learn'):
                        ni = b.get_input_vector()
                        b.learn(b.last_reward, ni, frame == 999)
                    b.last_reward = 0
        
        # Episode End: Collect Stats
        for arena in arenas:
            # Determine winner of this arena
            winner = max(arena.bots, key=lambda b: b.mass)
            for b in arena.bots:
                stats['rewards'][b.algo_name].append(b.total_reward)
                stats['kills'][b.algo_name].append(b.kills)
                stats['deaths'][b.algo_name].append(b.deaths)
                stats['survival'][b.algo_name].append(b.time_alive_current)
                stats['max_mass'][b.algo_name].append(b.max_mass_achieved)
                
                # Win Rate
                stats['wins'][b.algo_name].append(1.0 if b == winner else 0.0)
                
                # Accuracy (Kills per 100 frames alive as a proxy for efficiency)
                acc = (b.kills * 100 / max(1, b.time_alive_current))
                stats['accuracy'][b.algo_name].append(min(1.0, acc))
                
                # Reset bot for next episode
                b.total_reward = 0
                b.kills = 0
                b.deaths = 0
                b.time_alive_current = 0
                b.mass = 25
                b.max_mass_achieved = 25
        
        if (ep + 1) % 5 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"Episode {ep+1}/{EPISODES} | Time: {elapsed:.1f}m | Avg Kills (DQN): {np.mean(stats['kills']['DQN-Pro'][-NUM_ARENAS:]):.2f}")
            # Save models
            for b_type in [DQNBot, PPOBot, SniperBot, SurvivalBot]:
                # Save the first instance's model as representative
                data = arenas[0].bots[[type(x) for x in arenas[0].bots].index(b_type)].save_state()
                torch.save(data, os.path.join(TRAIN_SAVE_DIR, f"{b_type.__name__}_model.pth"))

    print(f"✅ Training complete. Duration: {(time.time() - start_time)/60:.1f} minutes.")
    plot_training_results(stats)

if __name__ == "__main__":
    train()
