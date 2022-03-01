"""
This script will execute the tracking of a person in a video stream
using an algorithm to control the movements of the drone.

We'll connect to the drone, get the video stream and check
whether a person is in the frame. If so, we'll use the algorithm
to control the drone, otherwise the drone will stay still.

*Before executing the script:*

Be sure that you connected the drone to the computer
"""
import cv2
from djitellopy import Tello

ARROW_KEYS = {
	'UP': 0,
	'DOWN': 1,
	'LEFT': 2,
	'RIGHT': 3
}
tello = Tello()
tello.connect()
print(f'Available battery: {tello.get_battery()}')
tello.streamon()
cap = tello.get_video_capture()


while True:
	ret, frame = cap.read()
	if not ret:
		break
	
	frame = cv2.resize(frame, (160, 90))
	cv2.imshow('Drone video frame', frame)

	if cv2.waitKey(1) == ARROW_KEYS['UP']:
		print(f'Pressed UP')
		tello.takeoff()

	if cv2.waitKey(1) == ARROW_KEYS['RIGHT']:
		print(f'Pressed RIGHT')
		tello.rotate_clockwise(60)

	if cv2.waitKey(1) == ARROW_KEYS['LEFT']:
		print(f'Pressed LEFT')
		tello.rotate_counter_clockwise(60)

	if cv2.waitKey(1) == ARROW_KEYS['DOWN']:
		print(f'Pressed DOWN')
		tello.land()
		tello.streamoff()
		break

cv2.destroyAllWindows()