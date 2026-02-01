# gamemodes.py – Modulare Spielmodi mit Standard-Interface für Meta-Learning
import math
import random
import torch
from entities import FoodPellet, Flag, Checkpoint, Resource, PuzzleElement, MAP_SIZE

# Modus-ID 0..9 für CRN/MSPN (Reihenfolge fest)
MODE_IDS = {
    "Classic Arena": 0, "Tag": 1, "Team Deathmatch": 2, "Capture the Flag": 3,
    "King of the Hill": 4, "Battle Royale": 5, "Infection": 6, "Resource Collector": 7,
    "Racing": 8, "Puzzle Cooperation": 9
}


class GameMode:
    """Basisklasse: Initialization, Update, Reward, Victory, Render-Specs."""
    def __init__(self, world):
        self.world = world
        self.name = "Abstract Mode"
        self.mode_id = 0
        self.episode_length = 3000
        self.allow_respawn = True
        self.victory_condition = "time_limit"
        self.reward_structure = "generic"

    def initialize(self):
        pass

    def update(self):
        pass

    def calculate_reward(self, bot):
        return 0

    def check_victory(self):
        if self.world.frame >= self.episode_length:
            return {"reason": "time_limit"}
        return None

    def get_render_specs(self):
        """Für Frontend: Zone, Zentrum, spezielle Darstellung."""
        return {"show_zone": False, "zone_radius": 0, "center_x": MAP_SIZE // 2, "center_y": MAP_SIZE // 2}

class ClassicArena(GameMode):
    def initialize(self):
        self.name = "Classic Arena"
        self.mode_id = MODE_IDS["Classic Arena"]
        self.victory_condition = "highest_mass_or_time"
        self.reward_structure = "mass_gain_kills_food"
        self.world.objects = []
        for _ in range(150): self.world.spawn_food()

class TagMode(GameMode):
    def initialize(self):
        self.name = "Tag"
        self.mode_id = MODE_IDS["Tag"]
        self.victory_condition = "most_tags_or_survive"
        self.reward_structure = "tagger_reward_runner_survival"
        self.world.objects = []
        self.world.food = []
        if not self.world.bots: return
        it_bot = random.choice(self.world.bots)
        for b in self.world.bots:
            if b == it_bot:
                b.team_id = 3 # Special "IT" team
                b.color = "#FF4500"
                b.status['is_it'] = True
            else:
                b.team_id = 0
                b.color = "#FFFFFF"
                b.status['is_it'] = False

    def update(self):
        it = next((b for b in self.world.bots if b.status.get('is_it')), None)
        if not it: return

        for b in self.world.bots:
            if b == it or b.died: continue
            dist = math.sqrt((it.x - b.x)**2 + (it.y - b.y)**2)
            if dist < (it.radius + b.radius):
                it.receive_reward(100)
                it.status['is_it'] = False
                it.team_id = 0; it.color = "#FFFFFF"
                b.receive_reward(-50)
                b.status['is_it'] = True
                b.team_id = 3; b.color = "#FF4500"
                break

    def calculate_reward(self, bot):
        if bot.status.get('is_it'): return -0.1
        return 0.1

class TeamDeathmatchMode(GameMode):
    def initialize(self):
        self.name = "Team Deathmatch"
        self.mode_id = MODE_IDS["Team Deathmatch"]
        self.victory_condition = "most_eliminations"
        self.reward_structure = "kills_teamwork_no_friendly_fire"
        self.world.objects = []
        self.world.food = []
        for i, b in enumerate(self.world.bots):
            if i < len(self.world.bots) // 2:
                b.team_id = 1; b.color = "#FF0000"
            else:
                b.team_id = 2; b.color = "#0000FF"

    def calculate_reward(self, bot):
        reward = 0
        for other in self.world.bots:
            if other != bot and other.team_id == bot.team_id and not other.died:
                dist = math.sqrt((bot.x - other.x)**2 + (bot.y - other.y)**2)
                if dist < 200: reward += 0.01
        return reward

class CaptureTheFlagMode(GameMode):
    def initialize(self):
        self.name = "Capture the Flag"
        self.mode_id = MODE_IDS["Capture the Flag"]
        self.victory_condition = "3_captures"
        self.reward_structure = "capture_return_defense"
        self.world.food = []
        self.world.objects = [
            Flag(1, 150, 150, 1),
            Flag(2, MAP_SIZE - 150, MAP_SIZE - 150, 2)
        ]
        for i, b in enumerate(self.world.bots):
            b.carrying_flag = False
            if i < len(self.world.bots) // 2:
                b.team_id = 1; b.color = "#FF0000"
            else:
                b.team_id = 2; b.color = "#0000FF"

    def update(self):
        for obj in self.world.objects:
            if isinstance(obj, Flag):
                if obj.carrier:
                    if obj.carrier.died:
                        obj.carrier.carrying_flag = False
                        obj.carrier = None
                        continue

                    obj.x, obj.y = obj.carrier.x, obj.carrier.y
                    base_x, base_y = (150, 150) if obj.carrier.team_id == 1 else (MAP_SIZE-150, MAP_SIZE-150)
                    if math.sqrt((obj.x-base_x)**2 + (obj.y-base_y)**2) < 100:
                        obj.carrier.receive_reward(500)
                        obj.carrier.carrying_flag = False
                        obj.carrier = None
                        obj.x, obj.y = (150, 150) if obj.team_id == 1 else (MAP_SIZE-150, MAP_SIZE-150)
                else:
                    for b in self.world.bots:
                        if not b.died and b.team_id != obj.team_id:
                            if math.sqrt((b.x-obj.x)**2 + (b.y-obj.y)**2) < b.radius + 20:
                                obj.carrier = b
                                b.carrying_flag = True
                                b.receive_reward(100)
                                break

    def calculate_reward(self, bot):
        if bot.carrying_flag: return 0.5
        # Shaped: Annäherung an gegnerische Flagge
        enemy_flag = next((obj for obj in self.world.objects if isinstance(obj, Flag) and obj.team_id != bot.team_id), None)
        if enemy_flag and not bot.carrying_flag:
            d = math.sqrt((bot.x - enemy_flag.x)**2 + (bot.y - enemy_flag.y)**2)
            return max(0, (2000 - d) / 20000.0)  # Dense reward
        return 0

class KingOfTheHillMode(GameMode):
    def initialize(self):
        self.name = "King of the Hill"
        self.mode_id = MODE_IDS["King of the Hill"]
        self.victory_condition = "longest_time_in_hill"
        self.reward_structure = "continuous_zone_bonus"
        self.world.objects = []
        self.world.food = []
        self.hill_x, self.hill_y, self.hill_r = MAP_SIZE//2, MAP_SIZE//2, 250

    def calculate_reward(self, bot):
        dist = math.sqrt((bot.x - self.hill_x)**2 + (bot.y - self.hill_y)**2)
        if dist < self.hill_r: return 1.0
        return -0.1

    def get_render_specs(self):
        return {"show_zone": True, "zone_radius": self.hill_r, "center_x": self.hill_x, "center_y": self.hill_y}

class BattleRoyaleMode(GameMode):
    def initialize(self):
        self.name = "Battle Royale"
        self.mode_id = MODE_IDS["Battle Royale"]
        self.victory_condition = "last_survivor"
        self.reward_structure = "survival_zone_damage"
        self.allow_respawn = False
        self.zone_radius = MAP_SIZE * 0.8
        self.world.objects = []
        for _ in range(75): self.world.spawn_food()

    def update(self):
        self.zone_radius = max(100, self.zone_radius - 0.5)
        center = MAP_SIZE // 2
        for b in self.world.bots:
            if b.died: continue
            dist = math.sqrt((b.x - center)**2 + (b.y - center)**2)
            if dist > self.zone_radius:
                b.mass -= 0.5
                b.receive_reward(-1)
                if b.mass <= 15: # MIN_BOT_MASS
                    self.world.respawn(b)

    def calculate_reward(self, bot):
        if not bot.died: return 0.2
        return 0

    def get_render_specs(self):
        return {"show_zone": True, "zone_radius": self.zone_radius, "center_x": MAP_SIZE // 2, "center_y": MAP_SIZE // 2}

class InfectionMode(GameMode):
    def initialize(self):
        self.name = "Infection"
        self.mode_id = MODE_IDS["Infection"]
        self.victory_condition = "all_infected_or_last_survivor"
        self.reward_structure = "zombie_conversion_survivor_time"
        self.world.objects = []
        self.world.food = []
        if not self.world.bots: return
        zombie = random.choice(self.world.bots)
        for b in self.world.bots:
            if b == zombie:
                b.team_id = 3 # Zombie
                b.color = "#00FF00"
                b.status['is_zombie'] = True
            else:
                b.team_id = 1 # Human
                b.color = "#FFFFFF"
                b.status['is_zombie'] = False

    def update(self):
        for b in self.world.bots:
            if b.team_id == 3:
                for other in self.world.bots:
                    if other.team_id == 1 and not other.died:
                        dist = math.sqrt((b.x - other.x)**2 + (b.y - other.y)**2)
                        if dist < (b.radius + other.radius):
                            other.team_id = 3
                            other.color = "#00FF00"
                            other.status['is_zombie'] = True
                            b.receive_reward(100)
                            other.receive_reward(-50)

    def calculate_reward(self, bot):
        if bot.team_id == 3: return 0.1
        return 0.2

class ResourceCollectorMode(GameMode):
    def initialize(self):
        self.name = "Resource Collector"
        self.mode_id = MODE_IDS["Resource Collector"]
        self.victory_condition = "most_resource_points"
        self.reward_structure = "gold_silver_bronze_value"
        self.world.objects = []
        self.world.food = []
        for _ in range(30):
            val = random.choice([1, 3, 5])
            col = "#CD7F32" if val == 1 else ("#C0C0C0" if val == 3 else "#FFD700")
            self.world.objects.append(Resource(random.randint(0, 1000000), random.uniform(50, MAP_SIZE-50), random.uniform(50, MAP_SIZE-50), val, col))

    def update(self):
        for b in self.world.bots:
            if b.died: continue
            for obj in self.world.objects[:]:
                if isinstance(obj, Resource):
                    dist = math.sqrt((b.x - obj.x)**2 + (b.y - obj.y)**2)
                    if dist < (b.radius + obj.radius):
                        b.receive_reward(obj.value * 20)
                        self.world.objects.remove(obj)
                        val = random.choice([1, 3, 5])
                        col = "#CD7F32" if val == 1 else ("#C0C0C0" if val == 3 else "#FFD700")
                        self.world.objects.append(Resource(random.randint(0, 1000000), random.uniform(50, MAP_SIZE-50), random.uniform(50, MAP_SIZE-50), val, col))

class RacingMode(GameMode):
    def initialize(self):
        self.name = "Racing"
        self.mode_id = MODE_IDS["Racing"]
        self.victory_condition = "all_checkpoints_in_order"
        self.reward_structure = "checkpoint_sequence_time"
        self.world.objects = []
        self.world.food = []
        for i in range(8):
            angle = math.radians(i * 45)
            x = MAP_SIZE // 2 + math.cos(angle) * 700
            y = MAP_SIZE // 2 + math.sin(angle) * 700
            self.world.objects.append(Checkpoint(i, x, y, i))
        for b in self.world.bots:
            b.status["next_checkpoint"] = 0

    def update(self):
        for b in self.world.bots:
            if b.died: continue
            next_cp_idx = b.status.get("next_checkpoint", 0)
            cp = next((obj for obj in self.world.objects if isinstance(obj, Checkpoint) and obj.sequence_num == next_cp_idx), None)
            if cp:
                dist = math.sqrt((b.x - cp.x)**2 + (b.y - cp.y)**2)
                if dist < (b.radius + cp.radius):
                    b.receive_reward(200)
                    b.status["next_checkpoint"] = (next_cp_idx + 1) % 8

    def calculate_reward(self, bot):
        next_cp_idx = bot.status.get("next_checkpoint", 0)
        cp = next((obj for obj in self.world.objects if isinstance(obj, Checkpoint) and obj.sequence_num == next_cp_idx), None)
        if cp:
            dist = math.sqrt((bot.x - cp.x)**2 + (bot.y - cp.y)**2)
            return (1000 - dist) / 10000.0
        return 0

class PuzzleCooperationMode(GameMode):
    def initialize(self):
        self.name = "Puzzle Cooperation"
        self.mode_id = MODE_IDS["Puzzle Cooperation"]
        self.victory_condition = "all_plates_activated_together"
        self.reward_structure = "joint_activation"
        self.world.objects = [
            PuzzleElement(1, MAP_SIZE//4, MAP_SIZE//2, "plate"),
            PuzzleElement(2, MAP_SIZE*3//4, MAP_SIZE//2, "plate")
        ]
        self.world.food = []

    def update(self):
        all_activated = True
        for obj in self.world.objects:
            if isinstance(obj, PuzzleElement):
                obj.activated = False
                for b in self.world.bots:
                    if b.died: continue
                    dist = math.sqrt((b.x - obj.x)**2 + (b.y - obj.y)**2)
                    if dist < (b.radius + obj.radius):
                        obj.activated = True; break
                if not obj.activated: all_activated = False

        if all_activated:
            for b in self.world.bots:
                if not b.died: b.receive_reward(10)

MODES = [ClassicArena, TagMode, TeamDeathmatchMode, CaptureTheFlagMode, KingOfTheHillMode,
         BattleRoyaleMode, InfectionMode, ResourceCollectorMode, RacingMode, PuzzleCooperationMode]
