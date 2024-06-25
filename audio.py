import speech_recognition as sr
import sys

 
r=sr.Recognizer()
filename=sys.argv[1]
with sr.AudioFile(filename) as source:
    
    audio=r.listen(source)
    try:
      
        print("You said : {}"+r.recognize_google(audio))
    except:
        print("Sorry could not recognize your voice")