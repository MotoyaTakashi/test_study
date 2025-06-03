# test

# Medical Consultation AI

import streamlit as st
import openai
import json
import numpy as np
import pandas as pd
import time
import requests

# API keys
openai.api_key = "openai"
DEEPSEEK_API_KEY = "deepseek"  # Replace with your DeepSeek API key

###Medical Knowledge###
# Symptom List
columns_dictionary_1 = [
    "chest pain", "dyspnea", "abdominal pain", "fever", "dizziness", 
    "headache", "altered consciousness", "palpitations", "convulsions", 
    "hematemesis", "bloody stool", "hematuria", "low back pain", 
    "back pain", "edema", "rash", "joint pain", "limb numbness", 
    "limb paralysis", "trauma", "insomnia", "rhinorrhea", "sore throat", 
    "cough", "fatigue"
]
# Addition Question List
next_question_map = {
    "abdominal pain": {
        1: [
            "Where exactly does the abdominal pain occur? (e.g., upper right abdomen, lower left abdomen, whole area)",
            "Could you describe the feeling of the pain? (Sharp pain, dull pain, stabbing pain, etc.)",
            "Is there any diarrhea or vomiting with this pain? If so, please tell me the frequency and number of times.",
            "If we rate the most severe pain you've ever experienced as 10 points, how would you rate the pain now?",
            "Does the pain continue all the time? Or does it come and go, getting stronger or weaker?",
            "When did you first start feeling this pain? (e.g., today morning, 2 days ago at night, 1 week ago, etc.)",
            "Have you recently eaten anything that might have caused a stomach upset? (e.g., spicy food, greasy food, etc.)",
            "When was your last bowel movement? Please tell me about the state of your bowel movement (hardness, color, and whether blood is mixed in).",
            "When was your last meal? Please also tell me what you ate.",
            "Do you know anyone else who has experienced similar symptoms?",
            "Have you ever had surgery on your abdomen? If so, please tell me what kind of surgery you had.",
            "Can you drink water? Are you able to drink water even if it's difficult?",
            "Do you feel any discomfort when you walk?"
        ],
        0: []
    },
    "insomnia": {
        1: ["When did you start having trouble sleeping?",
            "Do you have trouble falling asleep? Or do you wake up during the night?",
            "Is there anything that comes to mind as a cause of sleep deprivation? (Stress, anxiety, caffeine intake, etc.)",
            "When lying down, does the feeling of not being able to breathe get better or worse?",
            "Do you wake up to go to the bathroom at night? If so, please tell me about the number of times you do this.",
            "Have you ever thought about dying? (This question is very personal, so please answer honestly.)",
            "Are you having any trouble with your activities during the day?",
            # "Have you ever had trouble sleeping before? If so, please tell me about the situation and how you handled it.",
            # "Are you currently being treated for insomnia? (sleeping pills, lifestyle changes, psychological therapy, etc.)"
        ],
        0: []
    },
    "chest pain": {
        1: [
            "Did the pain start suddenly?",
            "Where exactly on your chest is it hurting? (midline, left, right, etc.)",
            "What kind of pain is it? (Tight, stabbing, burning, etc.)",
            "When did you first start feeling this kind of pain? (10 minutes ago, 3 days ago, etc.)",
            "If we rate the most severe pain you've ever experienced as 10 points, how would you rate the pain now?",
            "When you press on the area with your hand, does the pain get stronger or does it stay the same?",
            "How wide is the range of pain? Is it the size of a 10 yen coin, or is it the size of your hand?",
            "Does the pain get stronger when you climb stairs or sit down?",
            "Does it spread to your shoulders, arms, back, or neck?",
            "Do you feel any cold sweat or nausea?",
            "Do you feel any pain when you breathe deeply?",
        ],
        0: []
    },
    "dyspnea": {
        1: [
            "When did you first start feeling shortness of breath? (suddenly, gradually worsening, etc.)",
            "Do you feel shortness of breath even when you're at rest? (e.g., when lying down, during exercise, etc.)",
            "Do you feel shortness of breath when you can't sleep at night? (e.g., when lying down, during exercise, etc.)",
            # "How would you rate the severity of shortness of breath on a scale of 10?",
            "Do you feel any cough, fever, chest pain, or any other symptoms that come to mind?",
            "Do you have any underlying heart or lung diseases?",
            "Do you have any known triggers?"
        ],
        0: []
    },
    "fever": {
        1: [
            "When did you first start feeling feverish? (today morning, 1 month ago, etc.)",
            "What is the highest temperature you've had?",
            "Do you have any other symptoms besides fever? (cough, sore throat, runny nose, abdominal pain, diarrhea, rash, low back pain, etc.)",
            "Do you have any known triggers? (crowded places, international travel, surrounding infection situation, etc.)",
            "Does taking a fever-reducing medication help?"
        ],
        0: []
    },
    "dizziness": {
        1: [
            "What kind of dizziness do you experience? (dizzy feeling, floating feeling, standing up, etc.)",
            "Do you feel any difficulty moving your hands or feet?",
            "Do you have any headache?",
            "When did you first start feeling dizziness? (suddenly, gradually, etc.)",
            "When you feel dizzy, do you hear ringing in your ears or have any hearing problems?",
            "When you stand up or move your head, does your condition change?",
            "Have you ever experienced the same kind of dizziness before?",
        ],
        0: []
    },
    "headache": {
        1: [
            "Is the pain not as severe as the most severe pain in your life?",
            # "Where exactly on your head is it hurting? (frontal, posterior, lateral, etc.)",
            # "What kind of pain is it? (zicky-zacky, tight, heavy feeling, stabbing, etc.)",
            "When did it start? (10 minutes ago, 1 hour ago, 1 week ago, etc.)",
            "Did it suddenly get very painful? Did it gradually get worse over time?",
            "Do you have any difficulty speaking or understanding?",
            # "Does the headache come with nausea or vomiting, dizziness, sensitivity to light or sound, etc.?",
            # "Are you taking any headache medication?"
        ],
        0: []
    },
    "altered consciousness": {
        1: [
            "When did you first start feeling like you're not with it, or losing consciousness?",
            "Did anything happen before you lost consciousness that might have caused it? (strong pain, heat, shortness of breath, etc.)",
            "Did you have any seizures or incontinence when you lost consciousness?",
            "After recovering from altered consciousness, what was your state like? (immediately back to normal, stayed in a daze for a while, etc.)",
            "Have you ever experienced the same episode before?"
        ],
        0: []
    },
    "palpitations": {
        1: [
            "When do you feel palpitations? (when at rest, during exercise, under stress, etc.)",
            "How long do they last? (a few seconds, a few minutes, a few hours, etc.)",
            "Do they come with chest pain or shortness of breath or dizziness?",
            "Have you ever been told you have heart disease or arrhythmia?",
            "Do you have any caffeine intake or smoking habits?",
            "Have you ever experienced a loss of consciousness temporarily?"
        ],
        0: []
    },
    "convulsions": {
        1: [
            "Did you have a seizure? (if yes, please describe the situation)",
            "Do you have any other symptoms with the seizure? (e.g., loss of consciousness, incontinence, etc.)",
            "Do you feel any pain or discomfort during the seizure?",
            "Do you have any underlying medical conditions that might have caused the seizure?",
            "Have you ever had a similar seizure before?"
        ],
        0: []
    },
    "hematemesis": {
        1: [
            "When did you first notice that you were vomiting blood? (suddenly, gradually, etc.)",
            "How much blood was there? (how many cups of coffee grounds, small amount, etc.)",
            "What was the color and state of the blood? (bright red, dark, coffee-like, etc.)",
            "Was there any stomach pain or chest burn before vomiting blood?",
            "Have you ever been diagnosed with peptic ulcer, liver cirrhosis, or esophageal varices?"
        ],
        0: []
    },
    "bloody stool": {
        1: [
            "When did you first notice that you were having blood in your stool? (suddenly, gradually, etc.)",
            "What was the color and state of the blood? (bright red, dark, tar-like, etc.)",
            "Do you have any abdominal pain or diarrhea?",
            "Have you ever been diagnosed with hemorrhoids, ulcerative colitis, or colon polyps?",
            "Have you ever experienced the same symptoms before?"
        ],
        0: []
    },
    "hematuria": {
        1: [
            "When did you first notice that you were having blood in your urine? (suddenly, detected through a test, etc.)",
            "What was the color of your urine? (pink, reddish, brownish, etc.)",
            "Do you feel any pain or discomfort when you urinate? (burning sensation, residual urine, etc.)",
            "Do you have any other symptoms like fever, lower back pain, or swelling?",
            "Have you ever been told you have kidney disease or urinary tract problems?"
        ],
        0: []
    },
    "low back pain": {
        1: [
            "When did you first start feeling low back pain? (suddenly, chronic, etc.)",
            "What might be causing the pain? (carrying heavy objects, staying in the same position for a long time, etc.)",
            "What kind of pain is it? (sharp, dull, muscle pain, etc.)",
            "Does the pain change with different movements? (bending, twisting, sitting, standing up, etc.)",
            "Do you feel any pain or swelling in your legs or lower back?",
            "Do you have any bowel movements or urination problems?"
        ],
        0: []
    },
    "back pain": {
        1: [
            "Where exactly on your back is it hurting? (upper, middle, lower, etc.)",
            "When did it start? (from a fall, from exercise, from long desk work, etc.)",
            "What kind of pain is it? (dull, stabbing, burning, etc.)",
            "Do you feel any discomfort when you get up from bed or turn over?",
            "Do you feel any pain or swelling in your arms or chest?"
        ],
        0: []
    },
    "edema": {
        1: [
            "Where does the swelling appear on your body? (feet, face, hands, etc.)",
            "When did you start noticing that you have swelling? (when you wake up in the morning, in the evening, etc.)",
            "Does the swelling leave a mark when you press on it?",
            "Do you usually consume a lot of water or salt?",
            "Do you have any underlying medical conditions, such as heart disease, kidney disease, or liver disease, or are you taking any medication?"
        ],
        0: []
    },
    "rash": {
        1: [
            "Where does the rash appear on your body? (face, arms, trunk, etc.)",
            "When did you first start noticing it? (suddenly, gradually)",
            "What is the shape and characteristic of the rash? (red spots, blister, scaly, etc.)",
            "Do you feel any itching or burning, or any heat sensation?",
            "Have you ever experienced a similar rash before? Do you have any allergies?",
            "Have you ever been stung by a bug in a place like a mountain?"
        ],
        0: []
    },
    "joint pain": {
        1: [
            "Which joint is hurting? (knee, wrist, finger, shoulder, etc.)",
            "When did it start? (acute or chronic)",
            "What is the nature and characteristic of the pain? (zicky-zacky, swelling, heat, stiffness, etc.)",
            "When does the pain get stronger? (when you start moving, etc.)",
            "Have you ever been told you have a joint injury or rheumatism?"
        ],
        0: []
    },
    "limb numbness": {
        1: [
            "Where do you feel numbness? (hand, foot, one side, etc.)",
            "When did it start? Did you have any reason for it?",
            "Do you feel any pain or muscle weakness, or any sensory paralysis?",
            "Is it persistent or intermittent?",
            "Have you ever been told you have nerve or blood vessel disease, or hernia, etc.?"
        ],
        0: []
    },
    "limb paralysis": {
        1: [
            "Where do you feel paralysis? (right hand, left foot, etc.)",
            "When did you start feeling paralysis? (suddenly, gradually)",
            "Do you feel any pain or numbness with it?",
            "Is it progressing or is there a recovery trend?",
            "Have you ever been told you have a brain or nerve disease (stroke, etc.)?"
        ],
        0: []
    },
    "trauma": {
        1: [
            "When and how did you get hurt? (falling, traffic accident, sports, etc.)",
            "Which part of your body did you get hurt? (head, arm, leg, back, etc.)",
            "Do you have any symptoms like bleeding, pain, swelling, or deformity?",
            "Did you go to the hospital right after the injury? Did you receive first aid?",
            "Have you ever been hurt in the same place before?"
        ],
        0: []
    },

    "fatigue": {
        1: [
            "When did you first start feeling fatigue? (suddenly, gradually getting worse, etc.)",
            "What is the severity of your fatigue? (significantly affecting your daily life, etc.)",
            "Do you have any other symptoms like fever, loss of appetite, weight loss, etc.?",
            "Do you get enough sleep? Is your sleep cycle stable?",
            "Do you feel any stress or mental burden (work, interpersonal relationships, etc.)?"
        ],
        0: []
    },
    "rhinorrhea": {
        1: [
            "When did you first start having a runny nose? (suddenly, gradually, etc.)",
            "What is the color and viscosity of your nasal discharge? (clear, slightly yellow, sticky, with blood mixed, etc.)",
            "Do you have any other symptoms like sneezing, nasal congestion, sore throat, etc.?",
            "Do you have any underlying medical conditions like allergies or sinusitis?",
            "Do you know anyone else who has experienced similar symptoms?"
        ],
        0: []
    },
    "sore throat": {
        1: [
            "When did your throat start hurting? (suddenly, gradually)",
            "When do you feel the strongest pain?",
            "Do you have any other symptoms like swollen throat, redness, fever, nasal discharge, etc.?",
            "Have you ever repeated the same symptoms before? (e.g., pharyngitis, etc.)",
            "Is it not painful enough to swallow your saliva? (e.g., not painful enough to swallow saliva, etc.)",
            "Do you feel any pain when you try to swallow water?"
        ],
        0: []
    },
    "cough": {
        1: [
            "When did you first start coughing? (suddenly, gradually, etc.)",
            "What is the nature of your cough? (dry cough, mucus-containing cough, stronger at night, etc.)",
            "If you have mucus, what is the color and viscosity? (clear, yellowish, greenish, etc.)",
            "Do you have any fever, shortness of breath, or chest pain?",
            "Do you have any underlying medical conditions like smoking, allergies, or asthma?"
        ],
        0: []
    },
    "vomiting": {
    1: [
        "When did you first start vomiting? (suddenly, gradually increasing, etc.)",
        "Did you feel any pain in your chest or head?",
        "What is the frequency of vomiting? (1 time per day, every hour, etc.)",
        "Did you feel any nausea, abdominal pain, or chest burn before vomiting?",
        "What was the color and texture of what you vomited? (clear, white, yellow, green, brown, with blood mixed, etc.)",
        "Did you feel better after vomiting? Or does your bad mood continue?",
        "Do you have any diarrhea, fever, dizziness, etc. after vomiting?",
        "Do you have any dehydration symptoms (thirst, decreased urine output, etc.)?",
        "Do you eat well? Do you drink enough water?",
        "Have you ever experienced similar vomiting symptoms before? If so, what was the cause?"
    ],
    0: []
}
}
# Red Flag Sign
red_flag_sign_map = {
    "chest pain": [
        "cold sweat",
        "sudden onset",
        "pain scale >7",
        "dyspnea",
        "tightness when chest is squeezed",
        # "Shock Vital (blood pressure low, pulse fast)",
        # "low consciousness level"
    ],
    "dyspnea": [
        "SpO2<90%",
        "Tachypnea",
        "significant increase in breathing rate",
        "low consciousness level",
        "hypertension and shock symptoms"
    ],
    "abdominal pain": [
        "sudden severe pain",
        "abnormal vital signs (low blood pressure, fast pulse)",
        "board-like abdominal rigidity (strong muscular defense)",
        "blood in stool and hematemesis",
        "fever and chills"
    ],
    "fever": [
        # "fever over 39℃",
        "low consciousness level",
        # "subcutaneous bleeding rash/skin mucosa symptoms (serious infection)",
        # "significant changes in respiratory and circulatory dynamics",
        # "strong fatigue and weakness"
    ],
    "dizziness": [
        "sudden loss of consciousness or altered consciousness",
        "severe headache and vomiting",
        "language impairment and visual field abnormality",
        "difficulty walking and unilateral paralysis",
        "neck pain (possible intracranial vascular disease)"
    ],
    "headache": [
        "sudden severe headache (thunderclap headache)",
        "altered consciousness",
        "convulsions",
        "pain in neck and back (possible meningitis)",
        "symptom of nerve detachment (unilateral paralysis, sensory impairment, etc.)"
    ],
    "altered consciousness": [
        "not responding to pain or stimuli",
        "abnormal vital signs (especially respiratory rate and pulse)",
        "previous history of epilepsy or observation",
        "previous head injury or scar",
        "suspected poisoning by drugs or alcohol"
    ],
    "palpitations": [
        "chest pain",
        "loss of consciousness",
        "irregular pulse (arrhythmia)",
        "dyspnea",
        "shock vital signs (low blood pressure, fast pulse)"
    ],
    "convulsions": [
        "chest pain",
        "loss of consciousness",
        "seizure",
        "altered consciousness",
        "shock vital signs (low blood pressure, fast pulse)"
    ],
    "hematemesis": [
        "massive vomiting of blood",
        "black stool (possible bleeding from upper digestive tract)",
        "sudden severe anemia",
        "severe liver cirrhosis and peptic ulcer",
        "low consciousness level"
    ],
    "bloody stool": [
        "massive bleeding",
        # "shock vital signs (low blood pressure, fast pulse)",
        # "black stool (possible bleeding from upper digestive tract)",
        # "severe abdominal pain and rectal pain",
        "significant anemia"
    ],
    "hematuria": [
        "fresh red urine and wine-colored urine",
        "low back pain and lower abdominal pain",
        "difficulty urinating and frequent urination",
        "chills with fever (pyelonephritis, etc.)",
        "shock symptoms (massive hemorrhage)"
    ],
    "low back pain": [
        "acute severe pain (possible serious underlying condition other than low back pain)",
        "abnormal sensation and motor paralysis in lower limbs",
        "bladder-rectal dysfunction (urination/defecation disorder)",
        "fever and weight loss (possible infection/malignant tumor)",
        "abnormal vital signs"
    ],
    "back pain": [
        "sudden severe pain (possible dissection of aorta)",
        "difference in blood pressure between the two sides and abnormal vital signs",
        "spread of pain to chest and abdomen",
        "shock symptoms",
        "previous history of aortic aneurysm"
    ],
    "edema": [
        "sudden increase in weight",
        "dyspnea (possible heart failure)",
        "severe edema (systemic)",
        "decrease in urine output (possible kidney failure)",
        "clear signs of heart failure (sitting/breathing, etc.)"
    ],
    "rash": [
        "widespread and rapid increase",
        "blister formation and mucosal lesion (SJS, etc., serious skin disorder)",
        "severe pain and itching",
        "high fever and general fatigue",
        "shock symptoms (possible allergic reaction)"
    ],
    "joint pain": [
        "acute deformity and severe swelling",
        "significant limitation of range of motion",
        "severe redness and fever",
        "clear previous injury history",
        "symptoms of systemic disease (fatigue, weight loss, etc.)"
    ],
    "limb numbness": [
        "acute onset",
        "progressive paralysis",
        "disability in urination/defecation (possible spinal cord lesion)",
        "severe pain and sensory impairment",
        "previous history of intervertebral disc herniation and stroke"
    ],
    "limb paralysis": [
        "sudden onset (stroke, etc.)",
        "central nervous system symptoms (speech impairment, etc.)",
        "complications",
        "severe headache and dizziness",
        "previous history of arrhythmia and atrial flutter"
    ],
    "trauma": [
        "traumatic injury to the brain",
        "massive hemorrhage and open fracture",
        "unstable respiratory and circulatory dynamics",
        "suspected injury to the cervical spine (neck pain and limb paralysis)",
        "multiple injuries"
    ],
    "insomnia": [
        # "prolonged (lasting for several weeks)",
        "psychological symptoms (delusions, hallucinations, depression, etc.)",
        "severe fatigue and suicidal thoughts",
        "signs of sleep apnea (clear snoring, cessation of breathing)",
        "excessive daytime sleepiness affecting daily life"
    ],
    "rhinorrhea": [
        "bloody nasal discharge",
        # "large amount or bad smell of purulent nasal discharge",
        # "face pain and fever (possible serious paranasal sinusitis)",
        # "long-term odor loss",
        # "suspected leakage of spinal fluid after injury"
    ],
    "sore throat": [
        "difficulty swallowing",
        "dyspnea",
        "39℃ or higher fever",
        "difficulty swallowing saliva (possible pharyngitis)",
        "swelling of submandibular lymph nodes"
    ],
    "cough": [
        "dyspnea",
        "massive blood-streaked sputum",
        "persistent for 3 weeks or more (chronic cough)",
        "complications with chest pain and abnormal vital signs",
        "high fever and weight loss (possible infection/tuberculosis)"
    ],
    "vomiting": [
        "severe deterioration",
        "unable to get up",
        "severe dyspnea",
        "altered consciousness and severe dizziness",
        "significant weight loss (possible serious disease or serious infection)"
    ]
}
# List of departments
depertment_list = ['Internal Medicine', 'Plastic Surgery', 'Surgery', 'Dermatology', 'Ophthalmology', 'ENT', 'Pediatrics', 'Obstetrics and Gynecology', 'Urology', 'Neurology', 'Psychiatry', 'Cardiology', 'Emergency Medicine',  'Dentistry', 'Oral Surgery', 'Respiratory Medicine', 'Circulatory System', 'Digestive System', 'Endocrine Metabolism', 'Kidney', 'Hematology', 'Rheumatology', 'Allergology']

### Function Definitions ###
# Display text character by character
def typewrite(text: str, speed=0.05):
    typed_text = ""
    message_placeholder = st.empty()
    for char in text:
        typed_text += char
        message_placeholder.markdown(typed_text)
        time.sleep(speed)

# using chat GPT 4o
def chat_to_gpt_4o(prompt):
    MODEL = "gpt-4o-2024-08-06"
    completion = openai.ChatCompletion.create(
        model=MODEL,
        temperature=0,
        top_p=0.5,
        messages=[
            {"role": "system", "content": "You are a great assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def chat_to_gpt_4o_temperature_0(prompt):
    MODEL = "gpt-4o-2024-08-06"
    completion = openai.ChatCompletion.create(
        model=MODEL,
        temperature=0,
        top_p=0.5,
        messages=[
            {"role": "system", "content": "You are a great assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def chat_to_deepseek(prompt):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.session_state.get('deepseek_api_key', DEEPSEEK_API_KEY)}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a great assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "top_p": 0.5
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 401:
            st.error("DeepSeek API key is invalid. Please enter the correct API key in the sidebar.")
            return None
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to DeepSeek API: {str(e)}\nPlease check your API key.")
        return None

def chat_to_deepseek_temperature_0(prompt):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.session_state.get('deepseek_api_key', DEEPSEEK_API_KEY)}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a great assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "top_p": 0.5
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 401:
            st.error("DeepSeek API key is invalid. Please enter the correct API key in the sidebar.")
            return None
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to DeepSeek API: {str(e)}\nPlease check your API key.")
        return None

# Modify the existing functions to use either GPT-4 or DeepSeek based on a parameter
def chat_with_model(prompt, model="gpt4", temperature=0):
    try:
        if model == "gpt4":
            if temperature == 0:
                return chat_to_gpt_4o_temperature_0(prompt)
            return chat_to_gpt_4o(prompt)
        elif model == "deepseek":
            if temperature == 0:
                return chat_to_deepseek_temperature_0(prompt)
            return chat_to_deepseek(prompt)
        else:
            raise ValueError(f"Unsupported model: {model}")
    except Exception as e:
        st.error(f"API error occurred: {str(e)}\nPlease check if your API key is correctly set.")
        return None

# extract symptom from patient's comment
def out_put_dictionary(patients_comment, columns_dictionary=columns_dictionary_1):
    prompt = f"""
    Instruction:
    You are a great doctor. Below is what the patient said.
    Please check if any of the symptoms in the symptom list are present in the patient's statement,
    and return them in dictionary (JSON) format with 0/1 for each symptom.
    Constrains:
    Even if it doesn't seem to fit any of the symptoms, please carefully examine and choose the most similar one. For example, stomach discomfort or menstrual pain is considered abdominal pain.

    # Patient's statement
    {patients_comment}

    # Symptom List
    {json.dumps(columns_dictionary)}

    # Constraints
    - Please only output dictionary.
    - Example: {{"abdominal pain": 0, "sore throat": 1}}
    - Please do not provide any explanations or articles.
    - Please do not use line breaks.
    """

    str_output = chat_with_model(prompt, model=st.session_state["selected_model"], temperature=0)
    if str_output is None:
        st.error("Failed to analyze symptoms. Please check your API key.")
        return None
    
    try:
        dict_output = json.loads(str_output)
        return dict_output
    except json.JSONDecodeError:
        st.error("Failed to parse response from API.")
        return None

def extract_additional_symptom(patients_comment, columns_dictionary=columns_dictionary_1):
    prompt = f"""
    You are a great doctor. Below is a questionnaire from the patient, with questions and answers provided.
    Please check if any of the symptoms in the symptom list are present in the patient's statement,
    and return them in dictionary (JSON) format with 0/1 for each symptom.

    # Patient's questionnaire
    {patients_comment}

    # Symptom List
    {columns_dictionary}

    # Constraints
    - Please only output dictionary.
    - Example: { "abdominal pain": 0, "sore throat": 1 }
    - Please do not provide any explanations or articles.
    - Please do not use line breaks.
    """

    str_output = chat_with_model(prompt, model=st.session_state["selected_model"], temperature=0)
    dict_output = json.loads(str_output)

    return dict_output

# extract next question from additional question list
def get_additional_question(symptom_dict):
    next_question=[]

    for symptom, question_dict in next_question_map.items():
        if symptom in symptom_dict:
            value = symptom_dict[symptom]
            if value in question_dict:
                next_question.extend(question_dict[value])
    return next_question

# get next question from patient's comment
def get_next_question(patients_comment):
    symptom_dict = out_put_dictionary(patients_comment)
    next_question = get_additional_question(symptom_dict)
    return next_question

# If the question is already mentioned in the patient's statement, remove the additional question
def create_case_dict(patients_comment, next_question):
    case_dict = {}
    for i in range(len(next_question)):
        prompt = f"""
        Please extract the answer to the question from the patient's statement and put it in dictionary format.
        If the patient's statement does not contain an answer to the question or the answer is not accurately provided, please put 0.
        Patient's statement: {patients_comment},
        Question: {next_question[i]},
        Constraints:
        If you are unsure about whether the answer is correct, please put 0.
        If 0 or a string is returned, please do not provide any explanations or articles.
        - Example: 0,
        - Example: Too much caffeine.
        - Please do not provide any explanations or articles.
        - Please do not use line breaks.
        """
        str_response = chat_with_model(prompt, model=st.session_state["selected_model"], temperature=0)
        if str_response is None:
            st.error("Failed to analyze answer to question. Please check your API key.")
            return None
        case_dict[next_question[i]] = str_response
    return case_dict

def make_question_and_dictionary(patients_comment, columns_dictionary=columns_dictionary_1):
    # First, extract
    symptom_dictionary = out_put_dictionary(patients_comment, columns_dictionary)
    if symptom_dictionary is None:
        return None, None
    
    # Create next question list
    next_question_list = get_additional_question(symptom_dictionary)
    if not next_question_list:
        return {}, symptom_dictionary
    
    # Check if there is already an answer in the patient's statement
    case_dict = create_case_dict(patients_comment=patients_comment, next_question=next_question_list)
    if case_dict is None:
        return None, None
        
    return case_dict, symptom_dictionary


# Create summary and confirm
def make_summary(query_anwer_dictionary):
    prompt = f'''We want to make sure that the content of the medical questionnaire filled out by the patient is correct.
    Please check if there are any mistakes or concerns in the following medical report, and if there are, please let us know.
    Medical report: {query_anwer_dictionary}
  　Constraints: Please start from "I have summarized your symptoms as follows" and proceed.'''
    return chat_with_model(prompt, model=st.session_state["selected_model"])

# Create final summary combining initial summary and patient's additional comments
def make_final_summary(summary, patients_additional_comment):
    prompt = f'''Please create a final summary based on the patient's complaint and the doctor's additional comments.
    Summary: {summary}
    Patient's additional comments: {patients_additional_comment}
    Constraints: Please use bullet points and do not use any list formatting.
    Please provide the information in a natural order.
    Please do not reduce or add any information.
    This is for the patient to see, so please explain it in a way that is clear and understandable for a middle school student.'''
    return chat_with_model(prompt, model=st.session_state["selected_model"])

# Extract Red Flag Signs
def extract_red_flag_signs(structured_symptom):
    symptom_list = [k for k, v in structured_symptom.items() if v == 1]
    red_flag_sign_list = []
    for symptom in symptom_list:
        red_flag_sign_list.append(red_flag_sign_map[symptom])
    return red_flag_sign_list

# Determine urgency based on the presence of high-risk signs
def evaluate_urgency(summary, red_flag_sign_list):
    prompt = f'''Based on the patient's summary and the signs of danger, we need to determine if it's urgent to call an ambulance.
    Please check if any of the indicators of high urgency are present.
    If there are any signs of danger, please consider it urgent.
    Patient summary: {summary}
    High urgency indicators: {red_flag_sign_list}
    Example of output if urgency is low: You are having chest pain, but it's not serious. The risk of danger is high if you have cold sweat and still have chest pain, etc.
    We don't think it's urgent to call an ambulance now, but if there are any changes, please let us know.
    Example of output if urgency is high: You are having chest pain, but the risk of danger is high if you have cold sweat and still have chest pain, etc.
    You are showing signs of cold sweat, which is a risk factor for heart attack, etc., so we recommend calling an ambulance immediately.'''
    return chat_with_model(prompt, model=st.session_state["selected_model"])

# Consider which department to visit
def make_decision(summary_ver2):
    prompt = f'''Based on the content of the medical questionnaire, please consider the underlying causes of the symptoms and think about 2-3 possible diseases.
    Also, please select the department to visit from the list of medical departments and tell us the next department to visit.
    Medical questionnaire: {summary_ver2}
    List of medical departments: {depertment_list}
    Constraints: Please use bullet points and do not exceed 100 characters.
    Please generate a clear and understandable article for the patient.
    We recommend limiting the number of recommended departments to 3.
    '''
    return chat_with_model(prompt, model=st.session_state["selected_model"])

def hospital_iwami_decision(summary, depertment_assessement):
    prompt=f'''You are a doctor, currently on night duty, and are deciding whether to admit a patient.
    Please check if the patient meets the criteria for admission based on the following assessment of the patient's summary and the recommended department.
    Patient summary: {summary}
    Assessment of recommended department: {depertment_assessement}
    Criteria for admission: Difficult for internal medicine patients.
    Example: I'm from Iwami Hospital, but we only accept patients with plastic surgery cases today.
'''
    return chat_with_model(prompt, model=st.session_state["selected_model"], temperature=0)

def hospital_watanabe_decision(summary, depertment_assessement):
    prompt=f'''You are a doctor, currently on night duty, and are deciding whether to admit a patient.
    Please check if the patient meets the criteria for admission based on the following assessment of the patient's summary and the recommended department.
    Patient summary: {summary}
    Assessment of recommended department: {depertment_assessement}
    Criteria for admission: Difficult for internal medicine patients. However, if the patient has low blood pressure, shock, or needs surgery, it may be difficult.
    Example: I'm from Watanabe Hospital, but the possibility of not needing surgery is high for patients with internal medicine conditions. However, if the patient has low blood pressure, shock, or needs surgery, we can discuss the possibility of admission.
'''
    return chat_with_model(prompt, model=st.session_state["selected_model"], temperature=0)

def hospital_kikuoka_decision(summary, depertment_assessement):
    prompt=f'''You are a doctor, currently on night duty, and are deciding whether to admit a patient.
    Please check if the patient meets the criteria for admission based on the following assessment of the patient's summary and the recommended department.
    Patient summary: {summary}
    Assessment of recommended department: {depertment_assessement}
    Criteria for admission: All emergency departments are open. If the patient has no fever and low blood pressure, we would like to receive treatment at other hospitals.
    Example: I'm from Kikuoka Hospital, but we are a tertiary medical facility, so we can accept any patient. However, we also optimize medical resources, so we would appreciate it if you could receive treatment at other hospitals for lighter cases.
'''
    return chat_with_model(prompt, model=st.session_state["selected_model"], temperature=0)

def hospital_kato_decision(summary, depertment_assessement):
    prompt=f'''You are a doctor, currently on night duty, and are deciding whether to admit a patient.
    Please check if the patient meets the criteria for admission based on the following assessment of the patient's summary and the recommended department.
    Patient summary: {summary}
    Assessment of recommended department: {depertment_assessement}
    Criteria for admission: Difficult for internal medicine and surgical patients. However, only for patients with mental illness.
    Example: I'm from Kato Hospital, but we only accept patients with mental illness today.
'''
    return chat_with_model(prompt, model=st.session_state["selected_model"], temperature=0)

def hospital_saku_decision(summary, depertment_assessement):
    prompt=f'''You are a doctor, currently on night duty, and are deciding whether to admit a patient.
    Please check if the patient meets the criteria for admission based on the following assessment of the patient's summary and the recommended department.
    Patient summary: {summary}
    Assessment of recommended department: {depertment_assessement}
    Criteria for admission: Difficult for internal medicine and surgical patients. However, only for patients with mild heart disease and no need for abdominal surgery.
    Example: I'm from Saku Hospital, but we can accept patients with internal medicine and no need for abdominal surgery. However, for patients needing abdominal surgery, it is difficult.
'''
    return chat_with_model(prompt, model=st.session_state["selected_model"], temperature=0)

### Main Processing ###


def main():
    st.title("Medical Consultation AI")
    st.text("An AI for accurate medical consultation.")

    # Session state management
    if "step" not in st.session_state:
        st.session_state["step"] = 0
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = None
    if "api_keys" not in st.session_state:
        st.session_state["api_keys"] = {
            "openai": "",
            "deepseek": ""
        }

    # Sidebar settings
    with st.sidebar:
        st.markdown("Hello")
        st.markdown("### Feedback")
        st.markdown("Please provide your feedback and suggestions below.")
        st.markdown("[Survey](https://forms.gle/MuRWMHM23wPwPAQH8)")
        st.markdown("[GitHub Issues](https://github.com/yusukewatanabe1208/test/issues)")
        st.markdown("---")
        
        st.subheader("AI Model Settings")
        
        # Model selection (dropdown)
        model_choice = st.selectbox(
            "Select AI Model",
            ["GPT-4", "DeepSeek"],
            index=0
        )
        
        # API key input based on selected model
        if model_choice == "GPT-4":
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state["api_keys"]["openai"]
            )
        else:
            api_key = st.text_input(
                "DeepSeek API Key",
                type="password",
                value=st.session_state["api_keys"]["deepseek"]
            )
        
        if st.button("Save Settings and Start"):
            if api_key:
                # Save model and API key
                st.session_state["selected_model"] = "gpt4" if model_choice == "GPT-4" else "deepseek"
                
                # Save API key
                if model_choice == "GPT-4":
                    st.session_state["api_keys"]["openai"] = api_key
                    openai.api_key = api_key
                else:
                    st.session_state["api_keys"]["deepseek"] = api_key
                    global DEEPSEEK_API_KEY
                    DEEPSEEK_API_KEY = api_key
                    st.session_state["deepseek_api_key"] = api_key
                
                st.session_state.step = 1
                st.rerun()
            else:
                st.error("Please enter an API key.")

    # Main content
    if st.session_state.step == 0:
        st.info("Please select an AI model and enter your API key in the sidebar.")

    elif st.session_state.step == 1:
        # Register first assistant message if not exists
        if "assistants_first_comment" not in st.session_state:
            model_name = "GPT-4" if st.session_state["selected_model"] == "gpt4" else "DeepSeek"
            st.session_state["assistants_first_comment"] = f"I am an AI using {model_name} for accurate medical consultation.\nWhat brings you in today? Please describe your symptoms in 10-100 characters."
            st.session_state["messages"].append({
                "role": "assistant",
                "content": st.session_state["assistants_first_comment"],
                "typed": False
            })

    # --- step=1 and later, existing code ---
    elif st.session_state.step == 1:
        # If there is no initial assistant message yet, register one
        if "assistants_first_comment" not in st.session_state:
            model_name = "GPT-4" if st.session_state["selected_model"] == "gpt4" else "DeepSeek"
            st.session_state["assistants_first_comment"] = f"私は{model_name}を使用した正確な問診をするAIです。\n今日はどうされましたか？お困りのことを10-100文字程度で教えてください。"
            st.session_state["messages"].append({
                "role": "assistant",
                "content": st.session_state["assistants_first_comment"],
                "typed": False
            })

    # --- Display all past logs ---
    # If already typed=True, display immediately; if typed=False, display as typewriter
    for idx, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                # Check "typed" flag to decide whether to display as typewriter or immediately
                if not msg.get("typed", False):
                    # Only animate if not already displayed
                    typewrite(msg["content"], speed=0.05)
                    # After displaying, set flag to True
                    st.session_state["messages"][idx]["typed"] = True
                else:
                    # If already animated, just display
                    st.write(msg["content"])
            else:
                # User message is also OK to display immediately
                st.write(msg["content"])

    # --- User input field ---
    user_input = st.chat_input("Please enter your message...")
    if user_input:
        # Save user's message and display
        st.session_state["messages"].append({
            "role": "user",
            "content": user_input,
        })
        with st.chat_message("user"):
            st.write(user_input)

        # --- Branch based on current step ---
        if st.session_state.step == 1:
            # step1: User's free-form symptom description
            # ---------------------------------------------------
            st.session_state["patients_first_comment"] = user_input

            # Add next assistant message
            assistant_text = (
                "Thank you for your response.\n"
                "Next, I'll ask you some additional questions about the symptoms you've described.\n"
                "Please wait a moment."
            )
            st.session_state["messages"].append({
                "role": "assistant",
                "content": assistant_text,
                "typed": False
            })

            # Generate case_dict and symptom_dictionary
            result = make_question_and_dictionary(
                patients_comment=user_input,
                columns_dictionary=columns_dictionary_1
            )
            
            if result is None:
                st.error("Failed to analyze symptoms. Please check your API key.")
                return
                
            case_dict, symptom_dictionary = result
            st.session_state["case_dict"] = case_dict
            st.session_state["symptom_dictionary"] = symptom_dictionary

            # Create medical questionnaire
            if case_dict:
                # Format the case dictionary for better readability
                formatted_dict = {}
                for question, answer in case_dict.items():
                    if answer == "0":
                        formatted_dict[question] = "Not answered yet"
                    else:
                        formatted_dict[question] = answer

                # Display the questionnaire in a more readable format
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Based on your statement, I'll ask you some specific questions about your symptoms. Please answer them one by one."
                })
                
                # Display the first unanswered question
                unanswered = [q for q, a in case_dict.items() if a == "0"]
                if unanswered:
                    st.session_state["current_question"] = unanswered[0]
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": st.session_state["current_question"]
                    })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Symptom analysis is complete. Let's move on to the next question."
                })

            # Next
            st.session_state.step = 2
            st.rerun()
        elif st.session_state.step == 2:
            case_dict = st.session_state.get("case_dict")
            if case_dict is None:
                st.error("Failed to retrieve medical data. Please try again from the beginning.")
                st.session_state.step = 0
                st.rerun()
                return
            
            # If there is no current question being displayed, take the next question and display it
            if "current_question" not in st.session_state or st.session_state["current_question"] is None:
                unanswered = [q for q, a in case_dict.items() if a == "0"]
                if not unanswered:
                    # If all questions have been answered, proceed to step4
                    st.session_state.step = 4
                    st.rerun()
                else:
                    # Set current_question to the first unanswered question
                    st.session_state["current_question"] = unanswered[0]
                    # Add message for display
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": f"Please answer the following question:\n{st.session_state['current_question']}",
                        "typed": False
                    })
                    st.rerun()
            else:
                # current_question already exists → If user input is provided, use it as the answer
                if user_input:
                    # Save user input as the answer to the current question
                    question_to_answer = st.session_state["current_question"]
                    case_dict[question_to_answer] = user_input
                    st.session_state["case_dict"] = case_dict

                    # Check if there is a next question
                    unanswered = [q for q, a in case_dict.items() if a == "0"]
                    if not unanswered:
                        # If all questions have been answered, proceed to step4
                        done_text = "Thank you for answering all the questions. I'll now summarize your responses. Please wait."
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": done_text,
                            "typed": False
                        })
                        st.session_state.step = 4
                        # Set current_question to None
                        st.session_state["current_question"] = None
                    else:
                        # Still unanswered → Display next question
                        st.session_state["current_question"] = unanswered[0]
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": f"Please answer the following question:\n{st.session_state['current_question']}",
                            "typed": False
                        })
                    st.rerun()

        elif st.session_state.step == 4:
            # step4: Final determination
            # ---------------------------------------------------
            patients_additional_comment = user_input
            summary_ver1 = st.session_state["patients_summary_ver1"]
            summary_ver2 = make_final_summary(summary_ver1, patients_additional_comment)

            # Final summary
            assistant_text = f"Taking your additional comments into account, we've summarized as follows:\n\n{summary_ver2}"
            st.session_state["messages"].append({
                "role": "assistant",
                "content": assistant_text,
                "typed": False
            })

            # Urgency
            red_flag_sign_list = extract_red_flag_signs(st.session_state["symptom_dictionary"])
            urgency = evaluate_urgency(summary_ver2, red_flag_sign_list)
            assistant_text2 = f"Urgency Determination Result: {urgency}"
            st.session_state["messages"].append({
                "role": "assistant",
                "content": assistant_text2,
                "typed": False
            })

            # Recommended department
            recommend_depertment = make_decision(summary_ver2)
            assistant_text3 = f"Recommended Department for Consultation: {recommend_depertment}"
            st.session_state["messages"].append({
                "role": "assistant",
                "content": assistant_text3,
                "typed": False
            })
            # Possible medical facilities
            iwami_decision = hospital_iwami_decision(summary_ver2, recommend_depertment)
            assistant_text3 = f"From Iwami Hospital (Google Reviews XX points): {iwami_decision}"
            st.session_state["messages"].append({
                "role": "assistant",
                "content": assistant_text3,
                "typed": False
            })
            # Possible medical facilities
            watanabe_decision = hospital_watanabe_decision(summary_ver2, recommend_depertment)
            assistant_text3 = f"From Watanabe Hospital (Google Reviews XX points): {watanabe_decision}"
            st.session_state["messages"].append({
                "role": "assistant",
                "content": assistant_text3,
                "typed": False
            })
            # Possible medical facilities
            kikuoka_decision = hospital_kikuoka_decision(summary_ver2, recommend_depertment)
            assistant_text3 = f"From Kikuoka Hospital (Google Reviews XX points): {kikuoka_decision}"
            st.session_state["messages"].append({
                "role": "assistant",
                "content": assistant_text3,
                "typed": False
            })
            # Possible medical facilities
            kato_decision = hospital_kato_decision(summary_ver2, recommend_depertment)
            assistant_text3 = f"From Kato Hospital): {kato_decision}"
            st.session_state["messages"].append({
                "role": "assistant",
                "content": assistant_text3,
                "typed": False
            })
            # Possible medical facilities
            saku_decision = hospital_saku_decision(summary_ver2, recommend_depertment)
            assistant_text3 = f"From Saku Hospital): {saku_decision}"
            st.session_state["messages"].append({
                "role": "assistant",
                "content": assistant_text3,
                "typed": False
            })
            # End message
            final_msg = "That's all for now.\nPlease take care of yourself."
            st.session_state["messages"].append({
                "role": "assistant",
                "content": final_msg,
                "typed": False
            })

            st.session_state.step = 999
            st.rerun()

        elif st.session_state.step == 999:
            end_text = "Chat has ended. If you want to start over, please reload the page."
            st.session_state["messages"].append({
                "role": "assistant",
                "content": end_text,
                "typed": False
            })
            st.session_state.step = 1000
            st.rerun()

        elif st.session_state.step == 1000:
            pass


if __name__ == "__main__":
    main()
