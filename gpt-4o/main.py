import cv2
import base64
import os
from moviepy.editor import VideoFileClip
from openai import OpenAI

OPENAI_API_KEY = "your-openai-api-key"
VIDEO_PATH = "gpt-4o coding.mp4"
MODEL = "gpt-4o"

client = OpenAI(api_key=OPENAI_API_KEY)

def process_image(client, model):
    """Process an image and get a response from OpenAI."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful homework assistant that answers in yaml. Please help with my homework!"},
            {"role": "user", "content": [
                {"type": "text", "text": "What's the area of this triangle?"},
                {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/e/e2/The_Algebra_of_Mohammed_Ben_Musa_-_page_82b.png"}}
            ]}
        ],
        temperature=0.0,
    )
    print(response.choices[0].message.content)

def process_video(video_path, seconds_per_frame=2):
    """Extract frames and audio from a video file."""
    base64_frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame = 0

    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()

    print(f"Extracted {len(base64_frames)} frames")
    print(f"Extracted audio to {audio_path}")
    return base64_frames, audio_path

def summarize_video(client, model, frames):
    """Summarize video by sending frames to OpenAI."""
    filtered_frames = frames[::10]  # Send only every 10th frame
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a video summarizer. Please provide a summary of the video in YAML."},
            {"role": "user", "content": [
                "Here are the frames from the video.",
                *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, filtered_frames)
            ]},
        ],
        temperature=0,
    )
    print(response.choices[0].message.content)

def summarize_audio(client, model, audio_path):
    """Generate a summary of the audio transcription."""
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=open(audio_path, "rb"),
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are generating a transcript summary. Please create a summary of the provided transcription in yaml."},
            {"role": "user", "content": [
                {"type": "text", "text": f"The audio transcription is: {transcription.text}"}
            ]},
        ],
        temperature=0,
    )
    print(response.choices[0].message.content)

# Main execution
if __name__ == "__main__":
    process_image(client, MODEL)
    base64_frames, audio_path = process_video(VIDEO_PATH, seconds_per_frame=1)
    summarize_video(client, MODEL, base64_frames)
    summarize_audio(client, MODEL, audio_path)