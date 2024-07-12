# SPDX-License-Identifier: BSD-3-Clause

# flake8: noqa F401
from collections.abc import Callable
import math

import numpy as np

from vendeeglobe import (
    Checkpoint,
    Heading,
    Instructions,
    Location,
    Vector,
    config,
)
from vendeeglobe.utils import distance_on_surface


class Bot:
    """
    This is the ship-controlling bot that will be instantiated for the competition.
    """

    def __init__(self):
        self.team = "Galeon"  # This is your team name
        # This is the course that the ship has to follow
        self.adventure_time = 0
        self.course = [
            Checkpoint(21.210306994611223, -68.04859989857972, 5),
            Checkpoint(20.065931127231643, -74.1481107738384, 5),
            Checkpoint(9.506355070065034, -80.02362863091052, 5),
            Checkpoint(6.639683040069888, -78.90522764508958, 5),
            Checkpoint(6.7870862783949715, -80.8608611956101, 5),
            Checkpoint(6.7870862783949715, -80.8608611956101, 5),
            # first point
            Checkpoint(2.806318, -168.943864, 1950.0),
            Checkpoint(2.2614688264555594, 173.04590528139283, 5),
            # first after pacific
            Checkpoint(-0.0921654372694112, 132.45402991787185, 5),
            Checkpoint(-1.0078759837387682, 129.26939150280288, 5),
            Checkpoint(-2.9038261689957188, 127.44891222831968, 5),
            Checkpoint(-5.425670724081564, 127.07461761158417, 5),
            Checkpoint(-8.49556925204615, 125.23939387187525, 5),
            Checkpoint(-10.448048663542512, 120.98126234024876, 5),
            # second point
            Checkpoint(-5.66898412312321, 77.6746941312312, 50),
            Checkpoint(12.264261120963509, 51.245142337255054, 5),
            Checkpoint(12.229325931006983, 43.594915140088354, 5),
            Checkpoint(29.575756017299728, 32.548558842728404, 5),
            Checkpoint(31.637049573020285, 32.54430756794881, 5),
            Checkpoint(36.31844024735809, 14.638447507305397, 5),
            Checkpoint(38.027412428583354, 9.91961558807851, 5),
            Checkpoint(35.74023411130395, -6.6677929141895085, 5),
            Checkpoint(36.92097424602064, -9.491942340249679, 5),
            Checkpoint(38.783774691412404, -9.634937282486597, 5),
            Checkpoint(43.90554402658661, -9.277450109967123, 5),
            Checkpoint(
                latitude=config.start.latitude,
                longitude=config.start.longitude,
                radius=5,
            )
        ]
    def run(
        self,
        t: float,
        dt: float,
        longitude: float,
        latitude: float,
        heading: float,
        speed: float,
        vector: np.ndarray,
        forecast: Callable,
        world_map: Callable,
    ) -> Instructions:
        """
        This is the method that will be called at every time step to get the
        instructions for the ship.

        Parameters
        ----------
        t:
            The current time in hours.
        dt:
            The time step in hours.
        longitude:
            The current longitude of the ship.
        latitude:
            The current latitude of the ship.
        heading:
            The current heading of the ship.
        speed:
            The current speed of the ship.
        vector:
            The current heading of the ship, expressed as a vector.
        forecast:
            Method to query the weather forecast for the next 5 days.
            Example:
            current_position_forecast = forecast(
                latitudes=latitude, longitudes=longitude, times=0
            )
        world_map:
            Method to query map of the world: 1 for sea, 0 for land.
            Example:
            current_position_terrain = world_map(
                latitudes=latitude, longitudes=longitude
            )

        Returns
        -------
        instructions:
            A set of instructions for the ship. This can be:
            - a Location to go to
            - a Heading to point to
            - a Vector to follow
            - a number of degrees to turn Left
            - a number of degrees to turn Right

            Optionally, a sail value between 0 and 1 can be set.
        """
        # Initialize the instructions
        instructions = Instructions()

        # TODO: Remove this, it's only for testing =================
        current_position_forecast = forecast(
            latitudes=latitude, longitudes=longitude, times=2
        )
        # print(current_position_forecast)
        current_position_terrain = world_map(latitudes=latitude, longitudes=longitude)

        # ===========================================================
        # compute vector of current_position_forecast

        # Go through all checkpoints and find the next one to reach
        for ch in self.course:
            if ch.reached:
                continue


            # Compute the distance to the checkpoint
            dist = distance_on_surface(
                longitude1=longitude,
                latitude1=latitude,
                longitude2=ch.longitude,
                latitude2=ch.latitude,
            )

            # Check if the checkpoint has been reached
            if dist < ch.radius:
                ch.reached = True
            if ch.reached:
                continue

            # Consider slowing down if the checkpoint is close
            jump = dt * np.linalg.norm(speed)
            if dist < 2.0 * ch.radius + jump:
                instructions.sail = min(ch.radius / jump, 1)
            else:
                instructions.sail = 1.0


            instructions.location = Location(longitude=ch.longitude, latitude=ch.latitude)
            break
        return instructions