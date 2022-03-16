import unittest
from ModelReport.ModelReport import ModelReport
from DataHandler.DataHandler import DataHandler
import random
from time import process_time


class Test_ModelReport(unittest.TestCase):
    def test_CreateReport(self):
        modelName = "TestModel"
        modelCreator = "Tobias Rothlin"
        mlPrinciple = "Naive Bayes"
        refrences = {
            "Wikipedia": "https://en.wikipedia.org/wiki/Naive_Bayes_classifier",
            "Scikit-learn": "https://scikit-learn.org/stable/modules/naive_bayes.html",
        }
        algorithemDescription = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque id leo eu enim tempus rhoncus. Pellentesque fringilla at mi sit amet rutrum. Donec vehicula metus urna, maximus eleifend nibh consequat in. Mauris nunc ipsum, tristique non ligula id, tempor luctus turpis. Sed vehicula, ex id eleifend accumsan, augue nisi egestas mi, vel pretium justo magna quis erat. Donec et felis felis. Maecenas ut nunc ut urna sodales scelerisque. Sed sollicitudin facilisis sapien pharetra molestie. Donec accumsan nunc et dui aliquam, eget egestas ligula ullamcorper. Proin vitae tincidunt dolor. Vivamus imperdiet et velit vel laoreet. Donec nisi sem, rhoncus a orci id, ornare efficitur turpis. Mauris vitae ligula arcu. Aliquam felis neque, laoreet nec ante eu, dignissim tristique mi. Vivamus vitae lacus eu diam ultricies finibus at at libero."""
        graphicPath = "/Users/tobiasrothlin/Documents/BachelorArbeit/SentimentAnalysis/ROC_curves.png"
        graphicDescription = "The ROC curve of a naie Bayes Classifiere"

        # Using all Data as TrainingData
        myDatahandler = DataHandler()
        trainingSet = myDatahandler.getCategorieData("Location")

        # Creating the test Results
        mappingTable = {
            1: "Location",
            2: "Room",
            3: "Food",
            4: "Staff",
            5: "ReasonForStay",
            6: "GeneralUtility",
            7: "HotelOrganisation",
            8: "Unknown",
        }


        myModelReport = ModelReport(
            modelName,
            modelCreator,
            mlPrinciple,
            refrences,
            algorithemDescription,
            graphicPath,
            graphicDescription,
            "DataSetV1.2",
            "123420"
        )

        for m in range(100):
            testResults = []
            for i in range(1000):
                true = random.randint(1, 8)
                predicted = random.randint(1, 8)
                testResults.append(
                    [mappingTable[true], mappingTable[predicted]])
            myModelReport.addTestResults(testResults)
            random.seed(process_time())
            random.shuffle(trainingSet)
            myModelReport.addTrainingSet(trainingSet[0:-1000])


        myModelReport.createRaport(htmlDebug=True)

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
