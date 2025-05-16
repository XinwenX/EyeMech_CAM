from jetson_utils import videoSource, videoOutput, cudaToNumpy

# Set up the camera source and display output
## Customize the camera pipeline if needed
cam_id = 0
cam_width, cam_height, cam_framerate = 3280, 1848, 28
args = [f"--input-width={cam_width}", f"--input-height={cam_height}", f"--input-rate={cam_framerate}"]
## Use the custom pipeline if needed
camera = videoSource("csi://0", argv=args)

display = videoOutput("display://0")

while display.IsStreaming():
    img = camera.Capture()
    display.Render(img)
    display.SetStatus("FPS:{:.1f}".format(display.GetFrameRate()))