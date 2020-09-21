from __future__ import print_function
import nltk
import keyboard
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import datetime
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import speech_recognition as sr
import pyttsx3
import pytz  # timezone
import subprocess
stemmer = LancasterStemmer()

SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DAY_EXT = ["rd", "th", "st", "nd"]
CALCULATIONS = ["times", "divided", "minus", "plus"]


# loading words, labels, intents from trainingdata json file
with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
pattern_list = []  # list of all different patterns
pattern_tag = []  # how to classify these patterns - important in training model

# assigning data from json file to variables for DL module later use
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokenized_patterns = nltk.word_tokenize(pattern)  # spliting patterns as a list of words
        words.extend(tokenized_patterns)  # adding arguments from list 'tokenized_patterns' to list 'words'
        pattern_list.append(tokenized_patterns)  # adding from 'tokenized_patterns' to pattern_list
        pattern_tag.append(intent["tag"])  # adding intent tags to pattern tagss
    if intent["tag"] is not labels:  # if label doesn't exist in intent tags
        labels.append(intent["tag"])  # it's adding it to the list of intents

words = [stemmer.stem(w.lower()) for w in words if w != "?"]  # cutting
words = sorted(list(set(words)))  # set - removing duplicate elements, converting back to the list form and sorting
labels = sorted(labels)  # sorting labels - by default alphabetically
training = []  # it's gonna be list of 'bags of words'
output = []   # lists of 0's and 1's
out_empty = [0 for _ in range(len(labels))]

# creating 'bag' of words with all words in given pattern numerically
for i, j in enumerate(pattern_list):  # enumerate adds counter to an iterable and returns it
    bag = []
    stemmed_words = [stemmer.stem(w.lower()) for w in j]
    for w in words:
        if w in stemmed_words:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]  # copying empty output to output_row. Index finds the given element in a list and returns its position
    output_row[labels.index(pattern_tag[i])] = 1  # looking trough labels list for tags and setting it's value to 1 if it finds it
    training.append(bag)
    output.append(output_row)

training = numpy.array(training)  # tf requires numpy arrays to feed into
output = numpy.array(output)
with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, output, training), f)  # write these vars into pickle file to save it

# Willy's brain
tensorflow.reset_default_graph()  # reseting underlaying tf settings
net = tflearn.input_data(shape=[None, len(training[0])])  # input shape we're expecting for our model - how many wrds we have
net = tflearn.fully_connected(net, 8)  # adds hidden layer made of 8 neurons - every neuron conected to every neuron in neighbouring layer
net = tflearn.fully_connected(net, 8)  # another one  |  softmax allowes us to get the probability of outputs
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")  # output layer - lenght of amout of patterns
net = tflearn.regression(net)  # what is regression?

model = tflearn.DNN(net)  # DNN is a type of n. network - check out tensorflow types of networks
model.fit(training, output, n_epoch=300, batch_size=8, show_metric=True)  # passing training data to model
model.save("model.tflearn")  # batch means how many samples at once we're feeding to model

def bag_of_words(sentence, words):
    bag = [0 for _ in range(len(words))]  # creating empty bag of words
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    for wrd in sentence_words:  # looping trough words in sentence
        for i, w in enumerate(words):  # looping trough list of words
            if w == wrd:  # if word we're looping trough is in our sentence
                bag[i] = 1  # it's appending 1 to the bag
    return numpy.array(bag)

def delete():
    # removing old tensorflow data
    if os.path.exists("model.tflearn.data-00000-of-00001"):
        os.remove("model.tflearn.data-00000-of-00001")

    if os.path.exists("model.tflearn.index"):
        os.remove("model.tflearn.index")

    if os.path.exists("model.tflearn.meta"):
        os.remove("model.tflearn.meta")

    if os.path.exists("data.pickle"):
        os.remove("data.pickle")

    if os.path.exists("checkpoint"):
        os.remove("checkpoint")

def chat():
    print("Okay, I'm ready. You can talk to me now! (type exit to quit)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "exit":
            break
        results = model.predict([bag_of_words(inp, words)])  # predicting
        results_index = numpy.argmax(results)  # argmax returns most probable results
        tag = labels[results_index]  # assigning most probable result label to tag

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        print(random.choice(responses))

def speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[2].id)
    engine.setProperty('rate', 120)
    engine.setProperty('volume', 2.0)
    engine.say(text)
    print(text)
    engine.runAndWait()

def calibrate_microphone():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=5)
        r.dynamic_energy_threshold = True

def authenticate_google():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # if no valid creds available, let user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # saving creds for the future
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('calendar', 'v3', credentials=creds)
    return service

def get_events(day, service):
    date = datetime.datetime.combine(day, datetime.datetime.min.time())
    end_date = datetime.datetime.combine(day, datetime.datetime.max.time())
    utc = pytz.UTC
    date = date.astimezone(utc)
    end_date = end_date.astimezone(utc)
    events_result = service.events().list(calendarId='primary', timeMin=date.isoformat(), timeMax=end_date.isoformat(), singleEvents=True, orderBy='startTime').execute()
    events = events_result.get('items', [])

    if not events:
        answer = "I dont know about any events this day."
        speak(answer)
    else:
        answer = f"You have {len(events)} events on this day."
        speak(answer)

        for event in events:
            try:
                start = event['start'].get('dateTime', event['start'].get('date'))
                start_time = str(start.split("T")[1].split("-")[0])  # cuting data coming from calendar
                if int(start_time.split(":")[0]) < 12:
                    start_time = start_time + "am"
                else:
                    start_time = str(int(start_time.split(":")[0]) - 12) + start_time.split(":")[1]  # convert 24 hour time to 12
                    start_time = start_time + "pm"
                speak(event["summary"] + " at " + start_time)

            except Exception as e:
                print("Exeption occured: " + str(e))

def get_date(text):
    text = text
    today = datetime.date.today()

    if text.count("today") > 0:
        return today

    if text.count("tomorrow") > 0:
        return today + datetime.timedelta(days=+1)

    if text.count("yesterday") > 0:
        return today + datetime.timedelta(days=-1)

    day = -1
    day_of_week = -1
    month = -1
    year = today.year

    for word in text.split():
        if word in MONTHS:
            month = MONTHS.index(word) + 1
        elif word in DAYS:
            day_of_week = DAYS.index(word)
        elif word.isdigit():
            day = int(word)
        else:
            for ext in DAY_EXT:
                found = word.find(ext)
                if found > 0:
                    try:
                        day = int(word[:found])
                    except Exception:
                        pass

    if month < today.month and month != -1:  # if the month mentioned is before the current month set the year to the next
        year = year + 1

    if month == -1 and day != -1:  # if we didn't find a month, but we have a day
        if day < today.day:
            month = today.month + 1
        else:
            month = today.month

    # if we only found a day of the week
    if month == -1 and day == -1 and day_of_week != -1:
        current_day_of_week = today.weekday()
        dif = day_of_week - current_day_of_week

        if dif < 0:
            dif += 7
            if text.count("next") >= 1:
                dif += 7

        return today + datetime.timedelta(dif)

    if day != -1:
        return datetime.date(month=month, day=day, year=year)

def commands():
    print("exit")
    print("train")
    print("calibrate_microphone")
    print("talk")
    print("chat")

def save_note(text, title):
    file_name = str(title) + ".txt"
    with open(file_name, "w") as f:
        f.write(text)
    subprocess.Popen(["notepad.exe", file_name])

def check_time():
    now = datetime.datetime.now()
    hour = str(now.hour)
    minutes = str(now.minute)
    if int(hour) < 12:
        resp = str(hour) + " " + str(minutes) + " A M"
    else:
        hour = str(int(hour) - 12)
        resp = str(hour) + " " + str(minutes) + " P M"
    return(resp)

def open_program():
    resp = "Which program would you like me to open?"
    speak(resp)
    said = get_audio()
    if "music" in said:
        tidal = r"C:\Users\PC\AppData\Local\TIDAL\TIDAL.exe"  # r to convert string to raw string
        subprocess.Popen(tidal)
        speak("I'm opening Tidal for you")

def create_note():
    speak("Okay, how should I name this note?")
    print("I'm listening..")

    title = get_audio()
    print(title)

    speak("What should this note contain?")
    print("I'm listening..")
    note_text = get_audio()
    save_note(note_text, title)
    speak("I saved your note")

def answer(req):
    results = model.predict([bag_of_words(req, words)])[0]
    results_index = numpy.argmax(results)  # argmax returns most probable results
    tag = labels[results_index]  # assigning most probable result label to tag

    if results[results_index] < 0.4:
        answer = "I don't understand"
        speak(answer)

    else:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        answer = random.choice(responses)
        speak(answer)

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio)
            print("You: " + said)
        except Exception as e:
            print("Were you saying something? " + str(e))
    return said.lower()


def talk():
    while True:
        print("I'm listening..")
        req = get_audio()

        # R U N  A P P
        if "open program" in req:
            open_program()

        # S H U T  D O W N
        elif "go to sleep" in req:
            resp = "I already miss you. See you later my creator"
            speak(resp)
            break

        # S A V E  N O T E
        elif "create note" in req:
            create_note()

        # T I M E
        elif "what time is it" in req:
            resp = check_time()
            speak(resp)

        # C A L E N D A R
        elif get_date(req) is not None:
            date = get_date(req)
            print(date)
            get_events(date, SERVICE)

        # T A L K
        elif req is not None:
            answer(req)

        # N O  L O G I C  T R I G G E R E D
        else:
            print("I have no idea what you're talking about")


# I N T E R F A C E
while True:
    inp = input(">>>: ")
    SERVICE = authenticate_google()

    if inp.lower() == "exit":
        print("Shutting down the program...")
        break

    if inp.lower() == "calibrate":
        print("Calibrating microphone..")
        calibrate_microphone()
        print("Done.")

    if inp.lower() == "talk":
        talk()
        print("Talk ended.")

    if inp.lower() == "chat":
        chat()

    if inp.lower() == "delete":
        print("Removing old training data..")
        delete()
        print("Done.")

    if inp.lower() == "help":
        print("Available commands:")
        commands()
