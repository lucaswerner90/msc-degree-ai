#%%
import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
from model.visual_transformer_actor_critic import VisualEncoderActorCritic
from model.encoder_vit import EncoderViT

torch.manual_seed(42)

(IMAGE_WIDTH, IMAGE_HEIGHT) = (640, 360)
ACTIONS_PER_FRAME = 1

def get_image(image, point):
	current_point = int(point.item()*IMAGE_WIDTH)
	original_image = np.copy(image)
	image = cv2.circle(original_image, (current_point, int(IMAGE_HEIGHT/2)), radius=3, color=(0, 255, 255), thickness=10)
	
	image = cv2.putText(
		image,
		f"Predicted: {current_point}",
		(20,IMAGE_HEIGHT-50),
		cv2.FONT_HERSHEY_PLAIN,
		1,
		(0, 255, 255),
		1,
		cv2.LINE_AA
	)
	return image


def execute_action(point, action):
	distance = 20
	point = torch.Tensor([int(point.item() * IMAGE_WIDTH)])

	if action == "LEFT":
		point -= distance
	if action == "RIGHT":
		point += distance

	point = torch.clamp(point, 1, IMAGE_WIDTH-1)

	return point


experiment_name = 'ac_ac-vit-encoder_epoch_4'
encoder = EncoderViT()
model_file = f'checkpoints/{experiment_name}.pth'
video_output_dir = f'data/video_prediction/{experiment_name}'
video_output_path = f'{video_output_dir}/predicted_video.mp4'

model = VisualEncoderActorCritic()
model.load_state_dict(torch.load(model_file))
model.eval()


video_path = 'data/videos/03.MP4'
video = cv2.VideoCapture(video_path)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
progress_bar = tqdm(total=total_frames, desc=f'Predicting video frames from: {video_path}')

frames = []
point = torch.tensor([0.5],dtype=torch.float16)

while video.isOpened():
	ret, frame = video.read()
	if not ret:
		break
	
	image = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
	image = encoder(image)
	image = image.view(image.size(0), -1)
	logits, _ = model(image, point)
	_, action = model.select_action(logits)
	point = execute_action(point, action)
	frames.append(get_image(image, point))

	progress_bar.update(1)

if not os.path.exists(video_output_dir):
	os.makedirs(video_output_dir)

# write frames to video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
print(f'Writing predicted video to: {video_output_path}')
writer = cv2.VideoWriter(video_output_path, fourcc, 30, (IMAGE_WIDTH, IMAGE_HEIGHT))
for frame in frames:
	writer.write(frame)
writer.release()


