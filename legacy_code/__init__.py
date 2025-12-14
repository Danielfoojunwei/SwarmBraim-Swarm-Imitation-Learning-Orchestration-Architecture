"""
Perception modules for multi-view human pose estimation.

This package contains:
- cameras: ONVIF camera interface and multi-view synchronization
- pose_mmpose: MMPose/RTMW3D wrapper for 3D pose estimation
"""

# Handle imports gracefully when running outside of package context (e.g., pytest)
try:
    from .cameras import (
        OnvifCamera,
        MultiViewCameraRig,
        SynchronizedFrameSet,
        CameraCalibration,
        CameraFrame,
        triangulate_point,
        triangulate_points_ransac,
    )

    from .pose_mmpose import (
        RTMW3DInference,
        MultiViewConfig,
        Detection2D,
        create_pose_estimator,
        MockPoseDetector,
    )

    __all__ = [
        'OnvifCamera',
        'MultiViewCameraRig',
        'SynchronizedFrameSet',
        'CameraCalibration',
        'CameraFrame',
        'triangulate_point',
        'triangulate_points_ransac',
        'RTMW3DInference',
        'MultiViewConfig',
        'Detection2D',
        'create_pose_estimator',
        'MockPoseDetector',
    ]
except ImportError:
    # When imported outside of package context (e.g., during testing),
    # relative imports will fail. This is expected behavior.
    __all__ = []
