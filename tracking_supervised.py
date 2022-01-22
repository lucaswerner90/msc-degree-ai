"""
This script will execute the tracking of a person in a video stream
using an algorithm to control the movements of the drone.

We'll connect to the drone, get the video stream and check
whether a person is in the frame. If so, we'll use the algorithm
to control the drone, otherwise the drone will stay still.

*Before executing the script:*

Be sure that you connected the drone to the computer
"""

from drone.model import DroneModel


if __name__ == '__main__':
	drone = DroneModel()
