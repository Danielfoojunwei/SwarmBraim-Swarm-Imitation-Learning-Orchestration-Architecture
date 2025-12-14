"""Tests for recorder.py module."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from recorder import RobotObs, RobotAction, DemoStep, Episode


class TestRobotObs:
    """Tests for RobotObs dataclass."""

    def test_robot_obs_creation(self):
        """Test basic RobotObs creation."""
        obs = RobotObs(
            timestamp=0.0,
            joint_positions=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            joint_velocities=np.zeros(7),
            gripper_position=0.5,
        )
        assert obs.n_joints == 7
        assert obs.timestamp == 0.0
        assert obs.gripper_position == 0.5

    def test_to_vector(self):
        """Test conversion to vector."""
        obs = RobotObs(
            timestamp=0.0,
            joint_positions=np.array([0.1, 0.2, 0.3]),
            joint_velocities=np.array([0.4, 0.5, 0.6]),
            gripper_position=0.7,
        )
        vec = obs.to_vector()
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        np.testing.assert_array_almost_equal(vec, expected)

    def test_to_vector_no_gripper(self):
        """Test conversion to vector without gripper."""
        obs = RobotObs(
            timestamp=0.0,
            joint_positions=np.array([0.1, 0.2, 0.3]),
            joint_velocities=np.array([0.4, 0.5, 0.6]),
        )
        vec = obs.to_vector(include_gripper=False)
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        np.testing.assert_array_almost_equal(vec, expected)

    def test_vector_dim(self):
        """Test vector dimension calculation."""
        assert RobotObs.vector_dim(7, include_gripper=True) == 15
        assert RobotObs.vector_dim(7, include_gripper=False) == 14
        assert RobotObs.vector_dim(6, include_gripper=True) == 13


class TestRobotAction:
    """Tests for RobotAction dataclass."""

    def test_robot_action_creation(self):
        """Test basic RobotAction creation."""
        action = RobotAction(
            timestamp=0.0,
            joint_position_target=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            gripper_target=0.8,
        )
        assert action.n_joints == 7
        assert action.gripper_target == 0.8

    def test_to_vector(self):
        """Test conversion to vector."""
        action = RobotAction(
            timestamp=0.0,
            joint_position_target=np.array([0.1, 0.2, 0.3]),
            gripper_target=0.5,
        )
        vec = action.to_vector()
        expected = np.array([0.1, 0.2, 0.3, 0.5])
        np.testing.assert_array_almost_equal(vec, expected)

    def test_to_vector_no_gripper(self):
        """Test conversion to vector without gripper."""
        action = RobotAction(
            timestamp=0.0,
            joint_position_target=np.array([0.1, 0.2, 0.3]),
            gripper_target=0.5,
        )
        vec = action.to_vector(include_gripper=False)
        expected = np.array([0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(vec, expected)

    def test_vector_dim(self):
        """Test vector dimension calculation."""
        assert RobotAction.vector_dim(7, include_gripper=True) == 8
        assert RobotAction.vector_dim(7, include_gripper=False) == 7


class TestDemoStep:
    """Tests for DemoStep dataclass."""

    def create_obs(self, timestamp=0.0):
        """Helper to create RobotObs."""
        return RobotObs(
            timestamp=timestamp,
            joint_positions=np.random.randn(7),
            joint_velocities=np.random.randn(7),
            gripper_position=0.5,
        )

    def create_action(self, timestamp=0.0):
        """Helper to create RobotAction."""
        return RobotAction(
            timestamp=timestamp,
            joint_position_target=np.random.randn(7),
            gripper_target=0.5,
        )

    def test_demo_step_creation(self):
        """Test basic DemoStep creation."""
        step = DemoStep(
            step_index=0,
            obs=self.create_obs(),
            action=self.create_action(),
            episode_id="ep_001",
            task_id="task_001",
            env_id="env_001",
        )
        assert step.step_index == 0
        assert step.episode_id == "ep_001"
        assert step.is_valid
        assert step.quality_score == 1.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        step = DemoStep(
            step_index=0,
            obs=self.create_obs(),
            action=self.create_action(),
            episode_id="ep_001",
            task_id="task_001",
            env_id="env_001",
        )
        d = step.to_dict()
        assert 'step_index' in d
        assert 'obs' in d
        assert 'action' in d
        assert 'meta' in d
        assert d['meta']['episode_id'] == "ep_001"


class TestEpisode:
    """Tests for Episode dataclass."""

    def create_step(self, step_idx, timestamp):
        """Helper to create a DemoStep."""
        obs = RobotObs(
            timestamp=timestamp,
            joint_positions=np.random.randn(7),
            joint_velocities=np.random.randn(7),
            gripper_position=0.5,
        )
        action = RobotAction(
            timestamp=timestamp,
            joint_position_target=np.random.randn(7),
            gripper_target=0.5,
        )
        return DemoStep(
            step_index=step_idx,
            obs=obs,
            action=action,
            episode_id="ep_001",
            task_id="task_001",
            env_id="env_001",
        )

    def test_episode_creation(self):
        """Test basic Episode creation."""
        episode = Episode(
            episode_id="ep_001",
            task_id="task_001",
            env_id="env_001",
        )
        assert len(episode) == 0
        assert episode.duration == 0.0

    def test_append_step(self):
        """Test appending steps to episode."""
        episode = Episode(
            episode_id="ep_001",
            task_id="task_001",
            env_id="env_001",
        )
        for i in range(10):
            step = self.create_step(i, float(i))
            episode.append_step(step)

        assert len(episode) == 10
        assert episode.start_time == 0.0
        assert episode.end_time == 9.0
        assert episode.duration == 9.0

    def test_n_valid_steps(self):
        """Test counting valid steps."""
        episode = Episode(
            episode_id="ep_001",
            task_id="task_001",
            env_id="env_001",
        )
        for i in range(10):
            step = self.create_step(i, float(i))
            if i % 2 == 0:
                step.is_valid = False
            episode.append_step(step)

        assert episode.n_valid_steps == 5

    def test_get_obs_array(self):
        """Test getting observation array."""
        episode = Episode(
            episode_id="ep_001",
            task_id="task_001",
            env_id="env_001",
        )
        for i in range(5):
            step = self.create_step(i, float(i))
            episode.append_step(step)

        obs_array = episode.get_obs_array()
        assert obs_array.shape == (5, 15)  # 7 pos + 7 vel + 1 gripper

    def test_get_action_array(self):
        """Test getting action array."""
        episode = Episode(
            episode_id="ep_001",
            task_id="task_001",
            env_id="env_001",
        )
        for i in range(5):
            step = self.create_step(i, float(i))
            episode.append_step(step)

        action_array = episode.get_action_array()
        assert action_array.shape == (5, 8)  # 7 targets + 1 gripper

    def test_save_and_load(self):
        """Test saving and loading episode."""
        # Create episode with some steps
        episode = Episode(
            episode_id="ep_test",
            task_id="task_test",
            env_id="env_test",
            robot_type="test_robot",
        )
        for i in range(10):
            step = self.create_step(i, float(i))
            episode.append_step(step)

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_episode.npz"
            episode.save(path)

            # Load back
            loaded = Episode.load(path)

            assert loaded.episode_id == "ep_test"
            assert loaded.task_id == "task_test"
            assert loaded.env_id == "env_test"
            assert loaded.robot_type == "test_robot"
            assert len(loaded) == 10

    def test_empty_episode_save_load(self):
        """Test saving and loading empty episode."""
        episode = Episode(
            episode_id="ep_empty",
            task_id="task_test",
            env_id="env_test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty_episode.npz"
            episode.save(path)
            loaded = Episode.load(path)

            assert loaded.episode_id == "ep_empty"
            assert len(loaded) == 0
