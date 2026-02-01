# tests.py – Unit-Tests für Ais Meta-Learning Arena
import unittest
import torch
import numpy as np

from entities import BaseBot, FoodPellet, Flag, Checkpoint, Resource, PuzzleElement, MAP_SIZE
from mechanics import SpatialGrid, HeadlessArena
from gamemodes import (
    ClassicArena,
    TagMode,
    CaptureTheFlagMode,
    KingOfTheHillMode,
    BattleRoyaleMode,
)
from networks import SharedEncoder, CRN, MSPN, MCN
from bots import MetaBot, RuleBasedBot


class TestGameModes(unittest.TestCase):
    def test_classic_arena_initialization(self):
        """Classic Arena erzeugt Futter."""
        from app import GameWorld
        world = GameWorld()
        world.set_mode(ClassicArena)
        self.assertGreater(len(world.food), 0)
        self.assertEqual(world.current_mode.name, "Classic Arena")

    def test_tag_mode_role_assignment(self):
        """Tag Mode weist IT-Rolle zu."""
        from app import GameWorld
        world = GameWorld()
        world.bots = [RuleBasedBot(i) for i in range(5)]
        world.set_mode(TagMode)
        it_count = sum(1 for b in world.bots if b.team_id == 3)
        self.assertEqual(it_count, 1)

    def test_ctf_flag_spawning(self):
        """CTF spawnt zwei Flaggen."""
        from app import GameWorld
        world = GameWorld()
        world.set_mode(CaptureTheFlagMode)
        flags = [obj for obj in world.objects if isinstance(obj, Flag)]
        self.assertEqual(len(flags), 2)

    def test_king_of_hill_zone(self):
        """King of Hill hat definierte Zone."""
        from app import GameWorld
        world = GameWorld()
        world.set_mode(KingOfTheHillMode)
        self.assertIsNotNone(world.current_mode.hill_x)
        self.assertIsNotNone(world.current_mode.hill_y)
        self.assertGreater(world.current_mode.hill_r, 0)

    def test_mode_respawn_logic(self):
        """Battle Royale deaktiviert Respawn."""
        from app import GameWorld
        world = GameWorld()
        world.set_mode(BattleRoyaleMode)
        self.assertFalse(world.current_mode.allow_respawn)


class TestNetworks(unittest.TestCase):
    def test_shared_encoder_output_shape(self):
        """Encoder gibt korrekte Latent-Dimension."""
        encoder = SharedEncoder(input_dim=228, latent_dim=128)
        inp = torch.randn(1, 228)
        out = encoder(inp)
        self.assertEqual(out.shape, (1, 128))

    def test_crn_mode_prediction(self):
        """CRN gibt 10-dim Wahrscheinlichkeitsvektor."""
        encoder = SharedEncoder(input_dim=228, latent_dim=128)
        crn = CRN()
        seq = torch.randn(1, 100, 228)
        latent_seq = encoder(seq)
        mode_probs = crn(latent_seq)
        self.assertEqual(mode_probs.shape, (1, 10))
        self.assertAlmostEqual(mode_probs.sum().item(), 1.0, places=5)

    def test_crn_uncertainty(self):
        """CRN gibt Uncertainty zurück."""
        crn = CRN()
        seq = torch.randn(1, 100, 128)
        probs, unc = crn(seq, return_uncertainty=True)
        self.assertEqual(probs.shape, (1, 10))
        self.assertGreaterEqual(unc.item(), 0.0)
        self.assertLessEqual(unc.item(), 1.0)

    def test_mspn_actor_critic(self):
        """MSPN gibt Actor & Critic aus."""
        mspn = MSPN()
        latent = torch.randn(1, 128)
        probs, value = mspn(latent)
        self.assertEqual(probs.shape, (1, 12))
        self.assertEqual(value.shape, (1, 1))
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)


class TestBots(unittest.TestCase):
    def test_metabot_initialization(self):
        """MetaBot initialisiert korrekt."""
        encoder = SharedEncoder()
        crn = CRN()
        mspns = torch.nn.ModuleList([MSPN() for _ in range(10)])
        mcn = MCN()
        bot = MetaBot(0, encoder, crn, mspns, mcn)
        self.assertEqual(bot.algo_name, "Meta-Learner")
        self.assertEqual(len(bot.obs_buffer), 0)

    def test_metabot_decision_making(self):
        """MetaBot trifft Entscheidung."""
        encoder = SharedEncoder()
        crn = CRN()
        mspns = torch.nn.ModuleList([MSPN() for _ in range(10)])
        mcn = MCN()
        bot = MetaBot(0, encoder, crn, mspns, mcn)
        inp = np.random.randn(228).astype(np.float32)
        action = bot.decide(inp)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLessEqual(action, 11)

    def test_bot_vision_raycast(self):
        """Bot-Vision liefert korrekte Raycast-Dimensionen."""
        from app import GameWorld
        world = GameWorld()
        world.set_mode(ClassicArena)
        bot = RuleBasedBot(0)
        world.bots.append(bot)
        bot.sense(world)
        self.assertEqual(len(bot.ray_results), 24)
        self.assertEqual(len(bot.ray_results[0]), 9)


class TestTraining(unittest.TestCase):
    def test_headless_arena_step(self):
        """HeadlessArena führt Step aus."""
        bots = [RuleBasedBot(i) for i in range(3)]
        arena = HeadlessArena(ClassicArena, bots)
        results, victory = arena.step()
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 0)

    def test_spatial_grid_functionality(self):
        """SpatialGrid findet Entitäten."""
        grid = SpatialGrid()
        bot1 = RuleBasedBot(0)
        bot1.x, bot1.y = 500, 500
        bot2 = RuleBasedBot(1)
        bot2.x, bot2.y = 550, 550
        grid.add_entity(bot1)
        grid.add_entity(bot2)
        nearby = grid.get_nearby_entities(500, 500)
        self.assertIn(bot1, nearby)
        self.assertIn(bot2, nearby)


class TestEntities(unittest.TestCase):
    def test_base_bot_input_vector_length(self):
        """get_input_vector liefert 228 Dimensionen (24*9 + 12)."""
        bot = RuleBasedBot(0)
        bot.sense = lambda w: None  # dummy
        bot.ray_results = [[0.0] * 9 for _ in range(24)]
        from app import GameWorld
        world = GameWorld()
        world.current_mode = None
        vec = bot.get_input_vector(world)
        self.assertEqual(len(vec), 228)

    def test_mcn_output_shape(self):
        """MCN gibt 10-dim Gewichtung aus."""
        mcn = MCN()
        mode_probs = torch.randn(1, 10)
        mode_probs = torch.softmax(mode_probs, dim=-1)
        meta_stats = torch.randn(1, 5)
        out = mcn(mode_probs, meta_stats)
        self.assertEqual(out.shape, (1, 10))
        self.assertAlmostEqual(out.sum().item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
