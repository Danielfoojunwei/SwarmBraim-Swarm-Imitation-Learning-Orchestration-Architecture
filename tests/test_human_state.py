"""Tests for human_state.py module."""

import pytest
import numpy as np

from human_state import (
    EnvObject,
    Human3DState,
    DexterHandState,
    HumanState,
    fuse_to_human_state,
)


class TestEnvObject:
    """Tests for EnvObject dataclass."""

    def test_env_object_creation(self):
        """Test basic EnvObject creation."""
        pose = np.eye(4)
        obj = EnvObject(
            object_id="cup_001",
            class_id=1,
            class_name="cup",
            pose_world=pose,
            confidence=0.95,
        )
        assert obj.object_id == "cup_001"
        assert obj.class_id == 1
        assert obj.class_name == "cup"
        assert obj.confidence == 0.95
        assert obj.pose_world.shape == (4, 4)

    def test_env_object_with_bbox(self):
        """Test EnvObject with bounding box."""
        pose = np.eye(4)
        bbox = np.random.randn(8, 3)
        obj = EnvObject(
            object_id="box_001",
            class_id=2,
            class_name="box",
            pose_world=pose,
            bbox_3d=bbox,
        )
        assert obj.bbox_3d.shape == (8, 3)

    def test_env_object_invalid_pose(self):
        """Test that invalid pose raises assertion."""
        with pytest.raises(AssertionError):
            EnvObject(
                object_id="test",
                class_id=0,
                class_name="test",
                pose_world=np.eye(3),  # Wrong shape
            )


class TestHuman3DState:
    """Tests for Human3DState dataclass."""

    def test_human3d_creation(self):
        """Test basic Human3DState creation."""
        keypoints = np.random.randn(17, 3)
        confidence = np.random.rand(17)
        state = Human3DState(
            timestamp=1234567890.0,
            keypoints_3d=keypoints,
            keypoint_confidence=confidence,
        )
        assert state.timestamp == 1234567890.0
        assert state.keypoints_3d.shape == (17, 3)
        assert state.keypoint_confidence.shape == (17,)

    def test_get_joint(self):
        """Test retrieving named joints."""
        keypoints = np.zeros((17, 3))
        keypoints[16] = [1.0, 2.0, 3.0]  # r_wrist
        confidence = np.ones(17)
        state = Human3DState(
            timestamp=0.0,
            keypoints_3d=keypoints,
            keypoint_confidence=confidence,
        )
        r_wrist = state.get_joint('r_wrist')
        np.testing.assert_array_equal(r_wrist, [1.0, 2.0, 3.0])

    def test_wrist_properties(self):
        """Test wrist position properties."""
        keypoints = np.zeros((17, 3))
        keypoints[16] = [1.0, 0.0, 0.0]  # r_wrist
        keypoints[13] = [0.0, 1.0, 0.0]  # l_wrist
        state = Human3DState(
            timestamp=0.0,
            keypoints_3d=keypoints,
            keypoint_confidence=np.ones(17),
        )
        np.testing.assert_array_equal(state.wrist_right_position, [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(state.wrist_left_position, [0.0, 1.0, 0.0])

    def test_get_arm_direction(self):
        """Test arm direction computation."""
        keypoints = np.zeros((17, 3))
        keypoints[15] = [0.0, 0.0, 0.0]  # r_elbow
        keypoints[16] = [1.0, 0.0, 0.0]  # r_wrist
        state = Human3DState(
            timestamp=0.0,
            keypoints_3d=keypoints,
            keypoint_confidence=np.ones(17),
        )
        direction = state.get_arm_direction('right')
        np.testing.assert_array_almost_equal(direction, [1.0, 0.0, 0.0])


class TestDexterHandState:
    """Tests for DexterHandState dataclass."""

    def test_dexter_hand_creation(self):
        """Test basic DexterHandState creation."""
        finger_angles = np.zeros((5, 3))
        finger_abduction = np.zeros(4)
        wrist_quat = np.array([0.0, 0.0, 0.0, 1.0])
        hand = DexterHandState(
            timestamp=0.0,
            side='right',
            finger_angles=finger_angles,
            finger_abduction=finger_abduction,
            wrist_quat_local=wrist_quat,
        )
        assert hand.side == 'right'
        assert hand.finger_angles.shape == (5, 3)
        assert not hand.is_calibrated

    def test_invalid_finger_angles(self):
        """Test that invalid finger angles raise assertion."""
        with pytest.raises(AssertionError):
            DexterHandState(
                timestamp=0.0,
                side='right',
                finger_angles=np.zeros((4, 3)),  # Wrong shape
                finger_abduction=np.zeros(4),
                wrist_quat_local=np.array([0.0, 0.0, 0.0, 1.0]),
            )

    def test_finger_closure(self):
        """Test finger closure computation."""
        # Fully extended fingers
        finger_angles = np.zeros((5, 3))
        hand = DexterHandState(
            timestamp=0.0,
            side='right',
            finger_angles=finger_angles,
            finger_abduction=np.zeros(4),
            wrist_quat_local=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        closure = hand.get_finger_closure()
        assert closure.shape == (5,)
        np.testing.assert_array_almost_equal(closure, np.zeros(5))

    def test_grasp_aperture(self):
        """Test grasp aperture computation."""
        # Partially flexed fingers
        finger_angles = np.ones((5, 3)) * 0.5
        hand = DexterHandState(
            timestamp=0.0,
            side='right',
            finger_angles=finger_angles,
            finger_abduction=np.zeros(4),
            wrist_quat_local=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        aperture = hand.get_grasp_aperture()
        assert 0.0 <= aperture <= 1.0

    def test_has_contact_no_force_sensor(self):
        """Test contact detection without force sensor."""
        finger_angles = np.ones((5, 3)) * 1.5  # Highly flexed
        hand = DexterHandState(
            timestamp=0.0,
            side='right',
            finger_angles=finger_angles,
            finger_abduction=np.zeros(4),
            wrist_quat_local=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        # Without force sensor, estimates from closure
        assert isinstance(hand.has_contact(), bool)


class TestHumanState:
    """Tests for HumanState dataclass."""

    def create_body_state(self):
        """Helper to create a Human3DState."""
        return Human3DState(
            timestamp=0.0,
            keypoints_3d=np.random.randn(17, 3),
            keypoint_confidence=np.ones(17),
        )

    def create_hand_state(self, side='right'):
        """Helper to create a DexterHandState."""
        return DexterHandState(
            timestamp=0.0,
            side=side,
            finger_angles=np.zeros((5, 3)),
            finger_abduction=np.zeros(4),
            wrist_quat_local=np.array([0.0, 0.0, 0.0, 1.0]),
        )

    def test_human_state_creation(self):
        """Test basic HumanState creation."""
        body = self.create_body_state()
        state = HumanState(timestamp=0.0, body=body)
        assert state.body is body
        assert state.hand_right is None
        assert state.hand_left is None
        assert len(state.objects) == 0

    def test_human_state_with_hands(self):
        """Test HumanState with hand data."""
        body = self.create_body_state()
        hand_right = self.create_hand_state('right')
        state = HumanState(
            timestamp=0.0,
            body=body,
            hand_right=hand_right,
        )
        assert state.hand_right is hand_right

    def test_primary_object(self):
        """Test primary object selection."""
        body = self.create_body_state()
        obj = EnvObject(
            object_id="cup_001",
            class_id=1,
            class_name="cup",
            pose_world=np.eye(4),
        )
        state = HumanState(
            timestamp=0.0,
            body=body,
            objects=[obj],
            primary_object_id="cup_001",
        )
        assert state.primary_object is obj

    def test_primary_object_not_found(self):
        """Test primary object when ID doesn't exist."""
        body = self.create_body_state()
        state = HumanState(
            timestamp=0.0,
            body=body,
            primary_object_id="nonexistent",
        )
        assert state.primary_object is None


class TestFuseToHumanState:
    """Tests for fuse_to_human_state function."""

    def test_fuse_basic(self):
        """Test basic fusion."""
        human3d = Human3DState(
            timestamp=0.0,
            keypoints_3d=np.random.randn(17, 3),
            keypoint_confidence=np.ones(17),
        )
        result = fuse_to_human_state(human3d, None, None, [])
        assert isinstance(result, HumanState)
        assert result.body is human3d

    def test_fuse_auto_select_primary(self):
        """Test automatic primary object selection."""
        # Create human3d with wrists at known positions
        keypoints = np.zeros((17, 3))
        keypoints[16] = [0.0, 0.0, 0.0]  # r_wrist at origin
        keypoints[13] = [10.0, 0.0, 0.0]  # l_wrist far away
        human3d = Human3DState(
            timestamp=0.0,
            keypoints_3d=keypoints,
            keypoint_confidence=np.ones(17),
        )

        # Create object near right wrist
        pose = np.eye(4)
        pose[:3, 3] = [0.1, 0.0, 0.0]  # Near origin
        obj = EnvObject(
            object_id="nearby_cup",
            class_id=1,
            class_name="cup",
            pose_world=pose,
        )

        result = fuse_to_human_state(human3d, None, None, [obj])
        assert result.primary_object_id == "nearby_cup"
