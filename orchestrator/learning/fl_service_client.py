"""
Federated Learning Service Client

Replaces direct Flower FL integration with API calls to the unified
SwarmBridge federated learning coordinator.

This client allows the orchestrator to:
- Trigger training rounds
- Monitor training progress
- Apply aggregated model updates across the fleet
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class FLRoundStatus(str, Enum):
    """Status of a federated learning round"""
    PENDING = "pending"
    ACTIVE = "active"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FLRoundConfig:
    """Configuration for a federated learning round"""
    round_id: str
    learning_mode: str  # "single_actor", "multi_actor", "hybrid"
    privacy_mode: str  # "ldp", "dp_sgd", "he", "fhe", "secure_agg", "none"
    aggregation_strategy: str  # "mean", "trimmed_mean", "median", "krum", "secure_agg"
    min_participants: int = 2
    max_participants: int = 10
    timeout_seconds: int = 3600

    # Privacy parameters
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    noise_multiplier: Optional[float] = None
    clip_norm: Optional[float] = None

    # Multi-actor specific
    csa_base_id: Optional[str] = None
    num_actors: Optional[int] = None


@dataclass
class FLRoundStatus:
    """Status of a federated learning round"""
    round_id: str
    status: FLRoundStatus
    participants: List[str]
    started_at: datetime
    completed_at: Optional[datetime]
    aggregated_metrics: Dict[str, Any]


class FederatedLearningServiceClient:
    """
    Client for unified federated learning service (SwarmBridge)

    Replaces direct Flower FL integration with API calls to the
    centralized FL coordinator.
    """

    def __init__(
        self,
        swarm_bridge_url: str = "http://localhost:8083",
        timeout: int = 30,
    ):
        self.swarm_bridge_url = swarm_bridge_url.rstrip('/')
        self.timeout = timeout

        logger.info(f"FederatedLearningServiceClient initialized")
        logger.info(f"  SwarmBridge URL: {self.swarm_bridge_url}")

    def start_training_round(
        self,
        round_config: FLRoundConfig,
    ) -> Dict[str, Any]:
        """
        Start a new federated learning round

        Args:
            round_config: Round configuration

        Returns:
            Round start response with status and participants
        """
        logger.info(f"Starting FL round: {round_config.round_id}")

        payload = {
            "round_id": round_config.round_id,
            "learning_mode": round_config.learning_mode,
            "privacy_mode": round_config.privacy_mode,
            "aggregation_strategy": round_config.aggregation_strategy,
            "min_participants": round_config.min_participants,
            "max_participants": round_config.max_participants,
            "timeout_seconds": round_config.timeout_seconds,
            "epsilon": round_config.epsilon,
            "delta": round_config.delta,
            "noise_multiplier": round_config.noise_multiplier,
            "clip_norm": round_config.clip_norm,
            "csa_base_id": round_config.csa_base_id,
            "num_actors": round_config.num_actors,
        }

        try:
            response = requests.post(
                f"{self.swarm_bridge_url}/api/v1/rounds/start",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"FL round started successfully: {result.get('status')}")
            return result

        except requests.RequestException as e:
            logger.error(f"Failed to start FL round: {e}")
            raise

    def get_round_status(self, round_id: str) -> Dict[str, Any]:
        """
        Get status of a federated learning round

        Args:
            round_id: Round identifier

        Returns:
            Round status including participants and progress
        """
        try:
            response = requests.get(
                f"{self.swarm_bridge_url}/api/v1/rounds/{round_id}/status",
                timeout=self.timeout,
            )
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            logger.error(f"Failed to get round status: {e}")
            return {"round_id": round_id, "status": "unknown", "error": str(e)}

    def wait_for_round_completion(
        self,
        round_id: str,
        poll_interval: int = 10,
        max_wait: int = 3600,
    ) -> Dict[str, Any]:
        """
        Wait for a federated learning round to complete

        Args:
            round_id: Round identifier
            poll_interval: Seconds between status checks
            max_wait: Maximum seconds to wait

        Returns:
            Final round status
        """
        import time

        logger.info(f"Waiting for FL round {round_id} to complete")

        start_time = time.time()

        while True:
            status = self.get_round_status(round_id)

            if status.get("status") in ["completed", "failed"]:
                logger.info(f"FL round {round_id} finished with status: {status.get('status')}")
                return status

            elapsed = time.time() - start_time
            if elapsed > max_wait:
                logger.warning(f"FL round {round_id} timed out after {max_wait}s")
                return {"round_id": round_id, "status": "timeout"}

            time.sleep(poll_interval)

    def list_active_rounds(self) -> List[Dict[str, Any]]:
        """
        List all active federated learning rounds

        Returns:
            List of active rounds
        """
        try:
            response = requests.get(
                f"{self.swarm_bridge_url}/api/v1/rounds/active",
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            return result.get("rounds", [])

        except requests.RequestException as e:
            logger.error(f"Failed to list active rounds: {e}")
            return []

    def get_site_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about registered FL sites

        Returns:
            Site statistics including counts by type and mode
        """
        try:
            response = requests.get(
                f"{self.swarm_bridge_url}/api/v1/sites/statistics",
                timeout=self.timeout,
            )
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            logger.error(f"Failed to get site statistics: {e}")
            return {}

    def trigger_skill_update(
        self,
        skill_id: str,
        target_robots: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Trigger skill model update on robots after FL round completion

        Args:
            skill_id: Skill to update
            target_robots: List of robot IDs (None = all robots)

        Returns:
            Update status
        """
        logger.info(f"Triggering skill update for: {skill_id}")

        payload = {
            "skill_id": skill_id,
            "target_robots": target_robots or [],
        }

        try:
            response = requests.post(
                f"{self.swarm_bridge_url}/api/v1/skills/{skill_id}/update",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"Skill update triggered: {result.get('status')}")
            return result

        except requests.RequestException as e:
            logger.error(f"Failed to trigger skill update: {e}")
            raise

    def get_aggregated_metrics(
        self,
        round_id: str,
    ) -> Dict[str, Any]:
        """
        Get aggregated metrics from a completed round

        Args:
            round_id: Round identifier

        Returns:
            Aggregated metrics (loss, accuracy, etc.)
        """
        try:
            response = requests.get(
                f"{self.swarm_bridge_url}/api/v1/rounds/{round_id}/metrics",
                timeout=self.timeout,
            )
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            logger.error(f"Failed to get aggregated metrics: {e}")
            return {}

    def cancel_round(self, round_id: str) -> Dict[str, Any]:
        """
        Cancel an active federated learning round

        Args:
            round_id: Round identifier

        Returns:
            Cancellation status
        """
        logger.info(f"Cancelling FL round: {round_id}")

        try:
            response = requests.post(
                f"{self.swarm_bridge_url}/api/v1/rounds/{round_id}/cancel",
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"FL round cancelled: {result.get('status')}")
            return result

        except requests.RequestException as e:
            logger.error(f"Failed to cancel round: {e}")
            raise

    def monitor_training_progress(
        self,
        round_id: str,
    ) -> Dict[str, Any]:
        """
        Get real-time training progress for a round

        Args:
            round_id: Round identifier

        Returns:
            Training progress metrics
        """
        try:
            response = requests.get(
                f"{self.swarm_bridge_url}/api/v1/rounds/{round_id}/progress",
                timeout=self.timeout,
            )
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            logger.error(f"Failed to get training progress: {e}")
            return {"round_id": round_id, "progress": 0, "status": "unknown"}
