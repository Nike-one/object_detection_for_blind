import speech_recognition as sr
import pyttsx3

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)
engine.setProperty('rate',128)
engine.setProperty('volume',1)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def record_audio():
    r = sr.Recognizer()
    print("Starting recognition....")
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source=source,duration=0.5)
        audio_data = r.record(source,duration=5)
        print("Recognizing....")
        try:
            text = r.recognize_google(audio_data)
            print(text)
        except sr.UnknownValueError:
            print('Sorry, I did not get that')
        except sr.RequestError:
            print('Sorry, my speech service is down')
        return text

speak('Be independent')