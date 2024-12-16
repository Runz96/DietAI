from openai import AzureOpenAI
import os
import base64


class ChatApp:
    def __init__(self, model_name):
        api_key = ""
        api_base = ""
        api_type = "azure"
        api_version = "2023-05-15"
        # openai.api_version = '1106-Preview'
        
        self.client = AzureOpenAI(
            azure_endpoint = api_base, 
            api_key=api_key,  
            api_version=api_version
        )       
        
        self.model_name = model_name
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant to answer my questions."
            }
        ]
        
    def chat(self, message):
        self.messages.append({"role": "user", "content": message})
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            temperature=0.3,
            max_tokens=300
        )
        response = res.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response})
        return response

    def append_message(self, message):
        self.messages.append(message)
        

def open_img(image_path):
    # return a data url with base64 encoding
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return "data:image/png;base64," + encoded_string.decode('utf-8')


class Vision:
    def __init__(self, model_name):
        api_key = ""
        api_base = ""
        api_type = "azure"
        api_version = "2023-05-15"

        self.client = AzureOpenAI(
            azure_endpoint = api_base, 
            api_key=api_key,  
            api_version=api_version
        )        

        self.model_name = model_name
        self.messages = [
            {
                "role": "system",
                "content": "You are trained to interpret images about food and make responsible assumptions about them."
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
            temperature=0.3,
            max_tokens=200
        )
        response = res.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response})
        return response

    def append_message(self, message):
        self.messages.append(message)
        