""" Youtube Video Analysis Tool example code """

import os
import base64
import uuid
from pytube import YouTube
import cv2
from moviepy.editor import AudioFileClip
from openai import OpenAI


class YoutubeVideoAnalysis:
    """
    A class used to analyze YouTube videos.

    ...

    Attributes
    ----------
    openai_api_key : str
        The API key for OpenAI.
    vision_model_base_url : str
        The base URL for the vision model.
    vision_model : str
        The name of the vision model.
    whisper_model : str
        The name of the whisper model.

    Methods
    -------
    analyse_video(query: str, video_path: str) -> str:
        Analyzes the video and extracts insights.
    _process_video(video_path: str, seconds_per_frame=2):
        Extracts frames and audio from a video file.
    _transcribe(audio_path):
        Generates a summary of the audio transcription.
    """

    def __init__(
        self,
        vision_model_base_url: str,
        vision_model: str,
        openai_api_key: str,
        whisper_model: str = "base",
    ):
        """
        Constructs all the necessary attributes for the YoutubeVideoAnalysis object.

        Parameters
        ----------
            vision_model_base_url : str
                The base URL for the vision model.
            vision_model : str
                The name of the vision model.
            openai_api_key : str
                The API key for OpenAI.
            whisper_model : str
                The name of the whisper model.
        """
        self.vision_model_base_url = vision_model_base_url
        self.vision_model = vision_model
        self.llm = OpenAI(base_url=vision_model_base_url, api_key=openai_api_key)
        self.whisper_model = whisper_model

    def analyse_video(self, query: str, video_path: str) -> str:
        """
        Analyzes the video and extracts insights.

        Parameters
        ----------
            query : str
                The query to analyze.
            video_path : str
                The path to the video file.

        Returns
        -------
            str
                The analysis result.
        """
        try:
            base64_frames, audio_path = self._process_video(video_path)
            audio_transcription = self._transcribe(audio_path)
        except Exception as e:
            return f"Error processing video: {e}"

        try:
            response = self.llm.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You a video analysis tool. Please answer any questions on a given video.",
                    },
                    {
                        "role": "user",
                        "content": [
                            f"This is the query: {query}"
                            f"This is the video's audio transcription: {audio_transcription}"
                            "Here are the frames from the video.",
                            *map(
                                lambda x: {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpg;base64,{x}",
                                        "detail": "low",
                                    },
                                },
                                base64_frames[::5],
                            ),
                        ],
                    },
                ],
                temperature=0,
            )
        except Exception as e:
            return f"Error creating completion: {e}"

        return response.choices[0].message.content

    def _process_video(self, video_path: str, seconds_per_frame=2):
        """
        Extracts frames and audio from a video file.

        Parameters
        ----------
            video_path : str
                The path to the video file.
            seconds_per_frame : int, optional
                The number of seconds per frame (default is 2).

        Returns
        -------
            list
                The list of base64-encoded frames.
            str
                The path to the audio file.
        """
        base64_frames = []

        # Define the directory for temporary files
        temp_dir = os.path.join(os.path.dirname(__file__), "temp_tool_files")

        # Make sure the directory exists
        os.makedirs(temp_dir, exist_ok=True)

        # Create a YouTube object
        yt = YouTube(video_path)

        # Generate a unique identifier for the video
        video_id = str(uuid.uuid4())

        # Download the video file
        yt.streams.first().download(output_path=temp_dir, filename=f"{video_id}.mp4")

        # Define the path for the temporary video file
        temp_file_path = os.path.join(temp_dir, f"{video_id}.mp4")

        # Update the video path to the temporary file path
        video_path = temp_file_path

        base_video_path, _ = os.path.splitext(video_path)

        try:
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

            # Extract audio with moviepy
            audio_path = f"{base_video_path}.mp3"
            audio_clip = AudioFileClip(video_path)
            audio_clip.write_audiofile(audio_path, bitrate="32k")
            audio_clip.close()
        except Exception as e:
            raise Exception(f"Error processing video: {e}")
        finally:
            # Once done, delete the video file
            if os.path.exists(video_path):
                os.remove(video_path)

        return base64_frames, audio_path

    def _transcribe(self, audio_path):
        """
        Generates a summary of the audio transcription.

        Parameters
        ----------
            audio_path : str
                The path to the audio file.

        Returns
        -------
            str
                The transcription text.
        """
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = self.llm.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                )
            return transcription.text
        finally:
            # Once done, delete the audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)


def example_usage():
    """Example usage of the YoutubeVideoAnalysis class."""
    openai_api_key = "your_openai_api_key_here"
    vision_model_base_url = "https://api.openai.com/v1"
    vision_model = "gpt-4o"

    youtube_video_analysis = YoutubeVideoAnalysis(
        openai_api_key=openai_api_key,
        vision_model_base_url=vision_model_base_url,
        vision_model=vision_model,
    )

    print(
        youtube_video_analysis.analyse_video(
            query="What is this video about and what sort of imagery is used?",
            video_path="https://www.youtube.com/watch?v=Cx5aNwnZYDc&ab_channel=GeospatialWorld",
        )
    )
    
if __name__ == "__main__":
    example_usage()