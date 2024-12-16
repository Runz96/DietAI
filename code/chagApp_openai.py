from openai import OpenAI
import base64
import dotenv
dotenv.load_dotenv()


def open_img(image_path):
    # return a data url with base64 encoding
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return "data:image/png;base64," + encoded_string.decode('utf-8')


class Vision:
    def __init__(self, model_name):
        self.client = OpenAI()  

        self.model_name = model_name
        self.messages = [
            {
                "role": "system",
                "content": "You are an expert at analyzing images with computer vision. I will present you with a picture of food, which might be placed on a plate, inside a spoon, or contained within different vessels. Your job is to accurately identify the food depicted in the image."
                # "content": "You are trained to interpret images about food and make responsible assumptions about them."
            }
        ]
        
    def chat(self, message, image_path):
        self.messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message,
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": image_path
                    },
                }
            ]
        })
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            temperature=0.01,
            max_tokens=600
        )
        response = res.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response})
        return response

    def append_message(self, message):
        self.messages.append(message)