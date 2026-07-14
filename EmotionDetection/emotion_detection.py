"""
This module provides a function to analyze the emotion of a given text
using the Watson NLP emotion analysis service.
"""

import json
import requests


def emotion_detector(text_to_analyse):
    """
    Sends text to the Watson NLP emotion analysis endpoint, parses the
    response, and returns the emotion scores along with the dominant emotion.

    Args:
        text_to_analyse (str): The text to analyze for emotion.

    Returns:
        dict: A dictionary of the form
            {
                'anger': anger_score,
                'disgust': disgust_score,
                'fear': fear_score,
                'joy': joy_score,
                'sadness': sadness_score,
                'dominant_emotion': '<name of the dominant emotion>'
            }
    """

    base_url = 'https://sn-watson-emotion.labs.skills.network'
    url = f'{base_url}/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    input_json = {"raw_document": {"text": text_to_analyse}}

    response = requests.post(url, json=input_json, headers=headers, timeout=10)

    response_dict = json.loads(response.text)
    emotion_scores = response_dict['emotionPredictions'][0]['emotion']

    anger_score = emotion_scores['anger']
    disgust_score = emotion_scores['disgust']
    fear_score = emotion_scores['fear']
    joy_score = emotion_scores['joy']
    sadness_score = emotion_scores['sadness']

    emotions = {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score,
    }
    dominant_emotion = max(emotions, key=emotions.get)
    return {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score,
        'dominant_emotion': dominant_emotion,
    }
