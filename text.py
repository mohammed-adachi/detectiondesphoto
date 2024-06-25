from gtts import gTTS
text=""
with open('text.txt','r') as file:
    for line in file:
        text+=line
speech = gTTS(text)
speech.save('textt.mp3')
print("Text to speech conversion done")