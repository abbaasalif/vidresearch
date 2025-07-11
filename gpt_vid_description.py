# this file is useless for the current task
from openai import OpenAI
import base64

client = OpenAI()

def get_video_description(video_path: str) -> str:
    with open(video_path, 'rb') as f:
        video_data = f.read()

    video_base64 = base64.b64encode(video_data).decode('utf-8')

    response = client.chat.completions.create(
        model="gpt-4o-video-preview",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this video."},
                {"type": "video", "video": video_base64}
            ]}
        ]
    )

    return response.choices[0].message.content

