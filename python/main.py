from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
from yolo_pipeline import *
from lane import *


def pipeline_yolo(img):

    img_undist, img_lane_augmented, lane_info = lane_process(img)
    output = vehicle_detection_yolo(img_undist, img_lane_augmented, lane_info)

    return output
def pipeline_svm(img):

    img_undist, img_lane_augmented, lane_info = lane_process(img)
    output = vehicle_detection_svm(img_undist, img_lane_augmented, lane_info)

    return output

if __name__ == "__main__":

    demo = 1  # 1:image (YOLO image), 2: video (YOLO Pipeline)

    if demo == 1:
        f = open("weight.txt", "r")
        i = 1
        for x in f:
        	print(x)
        	image = mpimg.imread(x.strip())
        	yolo_result = pipeline_yolo(image)
        	plt.imsave('examples/result' + str(i), yolo_result)
        	i = i + 1
# (2) SVM pipeline
        draw_img = pipeline_svm(image)
        fig = plt.figure()
        plt.imshow(draw_img)
        plt.title('svm pipeline', fontsize=30)
        plt.show()


elif demo == 2:
        # YOLO Pipeline
        video_output = 'examples/project_YOLO.mp4'
        clip1 = VideoFileClip("examples/project_video.mp4").subclip(30,32)
        clip = clip1.fl_image(pipeline_yolo)
        clip.write_videofile(video_output, audio=False)
else:
        # SVM pipeline
        video_output = 'examples/project_svm.mp4'
        clip1 = VideoFileClip("examples/project_video.mp4").subclip(30, 32)
        clip = clip1.fl_image(pipeline_svm)
        clip.write_videofile(video_output, audio=False)





