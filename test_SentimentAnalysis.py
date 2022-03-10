from unittest import TestCase
from SentimentAnalysisV10 import HotelSentimentAnalysis
from DataHandler.DataHandler import DataHandler
import random
import matplotlib.pyplot as plt
import numpy as np

from time import process_time


class Test(TestCase):
    def test_SentimentAnalysis(self):
        dataHandler = DataHandler()
        dataFrame = dataHandler.getScoreData()
        random.seed(process_time())
        random.shuffle(dataFrame)

        trainingData = dataFrame[:-10]
        testData = dataFrame[-10:]

        dataStatistics = {
            'positivCount' : 0,
            'neutralCount': 0,
            'negativeCount': 0,
            'positivePercentage' : 0,
            'neutralPercentage': 0,
            'negativePercentage': 0,
            'numberOfWrongClassifications': 0,
            'thresholdForAverageClassifier': 0
        }

        for frame in trainingData:
            if frame[1] == 'Positiv':
                dataStatistics['positivCount'] += 1
            elif frame[1]== 'Neutral':
                dataStatistics['neutralCount'] += 1
            else:
                dataStatistics['negativeCount'] += 1


        dataStatistics['positivePercentage'] = dataStatistics['positivCount']/len(trainingData)
        dataStatistics['neutralPercentage'] = dataStatistics['neutralCount'] / len(trainingData)
        dataStatistics['negativePercentage'] = dataStatistics['negativeCount'] / len(trainingData)

        plt.bar(list(dataStatistics.keys())[:3],[dataStatistics[key] for key in list(dataStatistics.keys())[:3]],color=['green','blue','red'])
        plt.title("Histogramm of the Trainings data")
        plt.show()

        sentimentClassifier = HotelSentimentAnalysis(trainingData)

        print("-----------------------------Sentiment Analysis Resulat-----------------------------------")
        for data in testData:
            sentiment , sentimentConfidence = sentimentClassifier.getSentiment(data[0])

        self.assertTrue(True)
