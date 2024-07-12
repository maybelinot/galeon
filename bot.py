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



class Fork:
    def __init__(self, opt1, opt2):
        self.opt1 = opt1
        self.opt2 = opt2
        self.selected = None
        self.reached = False

class Bot:
    """
    This is the ship-controlling bot that will be instantiated for the competition.
    """

    def __init__(self):
        self.team = "Galeon"  # This is your team name
        # This is the course that the ship has to follow
        fork_1 = Fork(
            opt1 = [
                Checkpoint(21.210306994611223, -68.04859989857972, 20),
                Checkpoint(20.065931127231643, -74.1481107738384, 20),
                Checkpoint(9.506355070065034, -79.82362863091052, 20),
                Checkpoint(6.639683040069888, -78.90522764508958, 20),
                Checkpoint(6.7870862783949715, -80.8608611956101, 20),
                Checkpoint(2.806318, -168.943864, 50.0),
            ],
            opt2 = [
                Checkpoint(latitude=43.797109, longitude=-11.264905, radius=50),
                Checkpoint(longitude=-29.908577, latitude=17.999811, radius=50),
                Checkpoint(latitude=-11.441808, longitude=-29.660252, radius=50),
                Checkpoint(longitude=-63.240264, latitude=-61.025125, radius=50),
                Checkpoint(2.806318, -168.943864, 50.0),
            ]
        )
        self.course = [
            fork_1,
            # Checkpoint(2.2614688264555594, 173.04590528139283, 5),
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

            if isinstance(ch, Fork):
                if ch.selected is None:
                    time1 = time_to_get_to_fork(forecast, ch.opt1, latitude=latitude, longitude=longitude)
                    time2 = time_to_get_to_fork(forecast, ch.opt2, latitude=latitude, longitude=longitude)
                    print(time1, time2)
                    if time1 < time2:
                        ch.selected = ch.opt1
                    else:
                        ch.selected = ch.opt2
                for ch in ch.selected:
                    if ch.reached:
                        continue
                    break

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
            # lat_range = np.arange(latitude, ch.latitude, ((latitude<ch.latitude)*2-1)*0.1)
            # long_range = np.arange(latitude, ch.latitude, ((longitude<ch.longitude)*2-1)*0.1)
            # lat_long_combinations = np.array(np.meshgrid(lat_range, long_range)).T.reshape(-1, 2)

            # # latitudes = lat_long_combinations[:, 0]
            # # longitudes = lat_long_combinations[:, 1]
            # sea_map = world_map(latitudes=lat_long_combinations[:,0], longitudes=lat_long_combinations[:,1])
            # # forecast_map = forecast(latitudes=latitudes, longitudes=longitudes, times=1)
            # # horizontal_wind_speed = forecast_map[0]
            # # vertical_wind_speed = forecast_map[1]
            # # calculate vector of wind speed contribution for every point in sea_map if ship is moving towards checkpoint
            # # wind_speed_match = 


            # # joined_map = np.multiply(df.wind_in_direction_to_ch.values, sea_map)
            # sea_map_2d = sea_map.reshape(len(lat_range), len(long_range))
            # wind_forecast = forecast(latitudes=lat_long_combinations[:,0], longitudes=lat_long_combinations[:,1], times=1)
            # wind_map_2d_horizontal = wind_forecast[0].reshape(len(lat_range), len(long_range))
            # wind_map_2d_vertical = wind_forecast[1].reshape(len(lat_range), len(long_range))


            
            # import ipdb; ipdb.set_trace()
            # implement weighted  A* algorithm to find the best path to the checkpoint 
            # and give lat long of the point where to go first
            # def find_next_point_using_eighted_A_star(sea_map_2d, ch):
            #     import numpy as np
            


            # point = a_star_search(sea_map_2d, ch)
            
            # LON, LAT = np.meshgrid(long_range, lat_range)
            # plt.figure(figsize=(10, 6))
            # plt.pcolormesh(LON, LAT, sea_map_2d, cmap='coolwarm', shading='auto')
            # plt.colorbar(label='Sea (1) vs Land (0)')
            # plt.xlabel('Longitude')
            # plt.ylabel('Latitude')
            # plt.title('Sea and Land Map')
            # save to file
            # import ipdb; ipdb.set_trace()
            # plt.savefig('sea_map.png')
            # plt.show()
            # import ipdb; ipdb.set_trace()

            # plot sea_map based on axis of latitudes and longitudes

            # create a small map from current position to the checkpoint
            # map = world_map(latitudes=[latitude, ch.latitude], longitudes=[longitude, ch.longitude])
            # # create map of forecast
            # forecast_map = world_map(latitudes=current_position_forecast[0], longitudes=current_position_forecast[1])
            # # multiply both map and forecast_map, replasing map 0-s with nans
            # joined_map = np.multiply(map, forecast_map)
            # # find the most optimal heading based on the map
            # import ipdb; ipdb.set_trace()

            
            break

        return instructions

def time_to_get_to_fork(forecast, fork, longitude, latitude):

    if len(fork) == 0:
        return 0
    next_point = fork[0]
    mid_point = (latitude + next_point.latitude) / 2, (longitude + next_point.longitude) / 2
    distance = distance_on_surface(
        longitude1=longitude,
        latitude1=latitude,
        longitude2=next_point.longitude,
        latitude2=next_point.latitude,
    )
    # normalised vector from current position to next point
    vector = (next_point.longitude - longitude) / distance, (next_point.latitude - latitude) / distance
    current_position_forecast = forecast(
        latitudes=mid_point[0], longitudes=mid_point[1], times=0
    )
    speed = abs(vector[0]*current_position_forecast[0]) + abs(vector[1]*current_position_forecast[1])
    # calculate speed based on wind speed vector and current/next location
    time = distance / speed
    time += time_to_get_to_fork(forecast, fork[1:], next_point.longitude, next_point.latitude)
    return time

# def heuristic(a, b):
#     # Manhattan distance on a square grid
#     return abs(a[0] - b[0]) + abs(a[1] - b[1])

# def wind_impact(current, neighbor, wind_map_2d_horizontal, wind_map_2d_vertical):
#     dx = neighbor[0] - current[0]
#     dy = neighbor[1] - current[1]
#     wind_horizontal_impact = wind_map_2d_horizontal[current[0], current[1]] * dy
#     wind_vertical_impact = wind_map_2d_vertical[current[0], current[1]] * dx
#     # Accumulate wind impact positively, assuming that going against the wind increases cost
#     return abs(wind_horizontal_impact) + abs(wind_vertical_impact)



# def a_star_search(sea_map_2d, wind_map_2d_horizontal, wind_map_2d_vertical):
#     start = (0, 0)  # Start position is always (0, 0)
#     goal = (sea_map_2d.shape[0] - 1, sea_map_2d.shape[1] - 1)  # Goal is the last element in the map
#     neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-way movement
#     close_set = set()
#     came_from = {}
#     gscore = {start: 0}
#     fscore = {start: heuristic(start, goal)}
#     open_set = [start]

#     while open_set:
#         current = min(open_set, key=lambda x: fscore.get(x, np.inf))
#         if current == goal:
#             data = []
#             while current in came_from:
#                 data.append(current)
#                 current = came_from[current]
#             return data[::-1]

#         open_set.remove(current)
#         close_set.add(current)
#         for i, j in neighbors:
#             neighbor = current[0] + i, current[1] + j
#             if 0 <= neighbor[0] < sea_map_2d.shape[0] and 0 <= neighbor[1] < sea_map_2d.shape[1]:
#                 if sea_map_2d[int(neighbor[0]), int(neighbor[1])] == 0:  # Assuming 1 is non-traversable
#                     continue
#             else:
#                 continue  # Out of bounds

#             wind_cost = wind_impact(current, neighbor, wind_map_2d_horizontal, wind_map_2d_vertical)
#             tentative_g_score = gscore[current] + 1 + wind_cost

#             if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, np.inf):
#                 continue

#             if tentative_g_score < gscore.get(neighbor, np.inf) or neighbor not in [i for i in open_set]:
#                 came_from[neighbor] = current
#                 gscore[neighbor] = tentative_g_score
#                 fscore[neighbor] = gscore[neighbor] + heuristic(neighbor, goal)
#                 if neighbor not in open_set:
#                     open_set.append(neighbor)

#     return False