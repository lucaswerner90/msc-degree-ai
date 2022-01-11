import cv2
from djitellopy import Tello

tello = Tello()

tello.connect()
print(f'Available battery: {tello.get_battery()}')
tello.set_video_bitrate(Tello.BITRATE_1MBPS)
tello.streamon()
cap = tello.get_video_capture()


def record_video():
	while True:
		ret, frame = cap.read()
		if not ret:
			print("Can't receive frame. Exiting ...")
			break
		# Our operations on the frame come here
		# Display the resulting frame
		cv2.imshow('frame', frame)


		if cv2.waitKey(1) == ord('q'):
			tello.streamoff()
			break

record_video()