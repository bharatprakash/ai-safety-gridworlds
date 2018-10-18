import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='IslandNavigation-v0',
    entry_point='gym_safety_gridworlds.envs:IslandNavigation'
)
