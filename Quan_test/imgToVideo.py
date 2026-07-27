import cv2
import os

# img_dir = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/Generated_image/PartyScene_long/frameBase"
# img_dir = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/Generated_image/PartyScene_long/new/sharedNoise_fixedBG_CGE_v0_2"


# img_dir = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/Generated_image/BasketballPass/sharedNoise_v0"
# img_dir = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/Generated_image/BasketballPass/frameBase"
# img_dir = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/PartyScene_512_backup/gt"
# img_dir = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/BasketballPass_512_backup/images"


img_dir = "/media/ssd1/ndquan/videoInpainting/code/BrushNet/Quan_test/results/gen_image/ParkScene/finetuning_ckpt3500/propainter_unreliable_bg_full"
# img_dir = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/Generated_image/BasketballPass/new/fixedBG_nulltext_checkpoint3500" 
# img_dir_0 = "/media/ssd1/ndquan/videoInpainting/code/BrushNet/examples/brushnet/dataset/test/ParkScene/gt" 

# img_dir_1 = "/media/ssd1/ndquan/videoInpainting/code/BrushNet/examples/brushnet/dataset/test/ParkScene/inputs" 

# img_dir_2 = "/media/ssd1/ndquan/videoInpainting/code/BrushNet/examples/brushnet/dataset/test/Traffic/gt" 

# img_dir_3 = "/media/ssd1/ndquan/videoInpainting/code/BrushNet/examples/brushnet/dataset/test/Traffic/inputs" 



# vid_output_path = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/video/party_scene_sharedNoise_v1.mp4"
# vid_output_path = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/PartyScene_512_backup/PartyScene_gt_new.mp4"

vid_output_path = "/media/ssd1/ndquan/videoInpainting/code/BrushNet/Quan_test/results/gen_video/ParkScene/finetuning/flowCompletion_v1.mp4"
# vid_output_path = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/video/sharedNoise_fixedBG_CGE_v0_2.mp4"
# vid_output_path_0 = "/media/ssd1/ndquan/videoInpainting/code/BrushNet/examples/brushnet/dataset/test/ParkScene/video/gt.mp4"
# vid_output_path_1 = "/media/ssd1/ndquan/videoInpainting/code/BrushNet/examples/brushnet/dataset/test/ParkScene/video/inputs.mp4"
# vid_output_path_2 = "/media/ssd1/ndquan/videoInpainting/code/BrushNet/examples/brushnet/dataset/test/Traffic/video/gt.mp4"
# vid_output_path_3 = "/media/ssd1/ndquan/videoInpainting/code/BrushNet/examples/brushnet/dataset/test/Traffic/video/inputs.mp4"
# vid_output_path = "/media/ssd1/ndquan/videoInpainting/code/BrushNet/Quan_test/results/video/BasketballPass/fixedBG_nulltext_checkpoint3500.mp4"


# vid_output_path = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/PartyScene_512_backup/video/party_scene_input.mp4"

# Get list of image files in the directory
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])




def frameToVideo(img_dir, vid_output_path, fps=50):

    # Read the first image to get dimensions
    first_img_path = os.path.join(img_dir, img_files[0])
    frame = cv2.imread(first_img_path)
    height, width = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4

    video = cv2.VideoWriter(vid_output_path, fourcc, fps, (width, height))  # fps 

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"Video saved to {vid_output_path}")

if __name__ == "__main__":

    # os.makedirs(os.path.dirname(vid_output_path_0), exist_ok=True)
    # os.makedirs(os.path.dirname(vid_output_path_1), exist_ok=True)
    # os.makedirs(os.path.dirname(vid_output_path_2), exist_ok=True)
    # os.makedirs(os.path.dirname(vid_output_path_3), exist_ok=True)

    # frameToVideo(img_dir_0, vid_output_path_0, fps=24)
    # frameToVideo(img_dir_1, vid_output_path_1, fps=24)
    # frameToVideo(img_dir_2, vid_output_path_2, fps=30)
    # frameToVideo(img_dir_3, vid_output_path_3, fps=30)

    frameToVideo(img_dir, vid_output_path, fps=30)