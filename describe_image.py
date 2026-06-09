from dotenv import load_dotenv
import os
import base64
import mimetypes
from typing import Optional
from openai import AzureOpenAI


load_dotenv()

API_Key = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_IMAGE_ENDPOINT") or os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_IMAGE_API_VERSION", "2025-01-01-preview")
# Vision model for image inputs — competition instructions require GPT-5-mini
VISION_MODEL = (
    os.getenv("AZURE_OPENAI_IMAGE_MODEL")
    or os.getenv("AZURE_OPENAI_VISION_MODEL")
    or "gpt-5-mini"
)

if not API_Key:
    raise RuntimeError("Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY in .env or environment.")
if not AZURE_ENDPOINT:
    raise RuntimeError(
        "Missing Azure OpenAI image endpoint. Set AZURE_OPENAI_IMAGE_ENDPOINT or AZURE_OPENAI_CHAT_ENDPOINT in .env or environment."
    )


def _image_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    data = open(path, "rb").read()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


_DEFAULT_PROMPT = (
    "Please provide a concise description of the above image in 2-3 sentences. "
    "Mention visible objects, prominent colors, and number of each main items (ex if the picture is full of tables, mention how many tables are there). "
    "Return only the description text."
)


def describe_image(
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    image_title: Optional[str] = None,
    model: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    """Return a description for the image at `image_path` or `image_url`.

    Args:
        image_path: Local file path — encoded as a data URL and sent inline.
        image_url:  Public URL — sent directly to the model.
        model:      Override the default vision model (AZURE_OPENAI_VISION_MODEL).
        prompt:     Override the default description prompt.
    """

    prompt = f"""
    Please provide a concise description of the image in 2-3 sentences.

    The image title is: "{image_title}".
    The title will provide hints of what's going on in the photos, mention keyword names, years, etc in your description.
    The title may be complicated with multiple keywords using capital letters to split, make not to direct copy and paste to your response.
    Important wording rules:
    - If the title contains "gallery", describe it as "a gallery" or "gallery-style display".
    - Do not say "it looks like a gallery wall".
    - Avoid uncertain phrases like "looks like", "appears to be", or "seems to be" unless necessary.
    - Mention visible objects, prominent colors, and the number of main items if clear.
    - Return only the description text.
    """

    active_model = model or VISION_MODEL
    active_prompt = prompt or _DEFAULT_PROMPT

    client = AzureOpenAI(
        base_url=f"{AZURE_ENDPOINT}/deployments/{active_model}",
        api_key=API_Key,
        api_version=AZURE_API_VERSION,
    )
    if image_path:
        img_ref = _image_to_data_url(image_path)
    elif image_url:
        img_ref = image_url
    else:
        raise ValueError("Provide either image_path or image_url")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_ref}},
                {"type": "text", "text": active_prompt},
            ],
        }
    ]

    resp = client.chat.completions.create(
        model=active_model,
        messages=messages,
        max_completion_tokens=1500,
    )

    return resp.choices[0].message.content or ""


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Describe an image using Azure OpenAI")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image-path", help="Local image file path to describe")
    g.add_argument("--image-url", help="Public image URL to describe")
    args = p.parse_args()

    desc = describe_image(image_path=args.image_path, image_url=args.image_url)
    print(desc)
