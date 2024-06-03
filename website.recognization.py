#!/usr/bin/env python
# coding: utf-8

# In[12]:


import speech_recognition as sr
import webbrowser
from bs4 import BeautifulSoup
import requests
import pyttsx3

recognizer = sr.Recognizer()

website_summary = """
Welcome to Nuevera Infotech, your one-stop destination for top-notch web creation, hosting, marketing, and business management solutions. We also specialize in project management and deliver inspiring corporate talks. Elevate your skills through our comprehensive corporate internships and training programs, all aimed at ensuring complete customer satisfaction. Discover excellence in the digital realm with our trusted services.
"""

def recognize_speech():
    with sr.Microphone() as source:
        print("Please ask a question or say 'open website' or 'tell me the summary'.")
        recognizer.adjust_for_ambient_noise(source) 
        audio = recognizer.listen(source, timeout=5)  

    try:
        question = recognizer.recognize_google(audio)
        print("You said:", question)

        response = get_response(question)
        print("Response:", response)

    except sr.UnknownValueError:
        print("Sorry, I could not understand your question.")
    except sr.RequestError as e:
        print("Sorry, there was an error with the speech recognition service: {0}".format(e))

def get_response(question):
    if "hello" in question.lower():
        return "Hello! How can I assist you today?"
    elif "how are you" in question.lower():
        return "I'm fine, thank you!"
    elif "open website" in question.lower():
        web_url = "https://www.nueverainfotech.com/"
        webbrowser.open(web_url)
        return f"Opening {web_url} in your web browser."
    elif "tell me the summary" in question.lower():
        return website_summary
    else:
        return "I'm not sure how to respond to that."

def speak_response(response):
    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()

if __name__ == "__main__":
    recognize_speech()
    response = get_response("tell me the summary")
    speak_response(response)





# In[ ]:




