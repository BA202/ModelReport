import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from datetime import datetime
import pdfkit
import os
import seaborn as sn
import pandas as pd
import platform


class ModelReport:
    def __init__(
        self,
        modelName,
        creatorName,
        MLPrinciple,
        dictOfReferences,
        algoDescription,
        descriptionGraphicPath="",
        graphicDescription="",
        datafile = None,
        randomSplitSeed = None
    ):
        """
        Creates a ModelReport object. Defines the Overview section of the model report.

        Parameters
        ----------
        modelName : str
            is the name of the Model.
        creatorName : str
            the creator of the Report.
        MLPrinciple : str
            the underlying ML principle of the Model.
        dictOfReferences : dict
            dict containing Links {'WebsiteName':'FullLink'}.
        algoDescription: str
            multiline string describing the algorithm.
        descriptionGraphicPath: str
            absolute file path to a img. Gets placed next to the algoDescription.
        graphicDescription: str
            short string describing the img.
        """
        self.__modelName = modelName
        self.__date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.__creatorName = creatorName
        self.__MLPrinciple = MLPrinciple
        self.__dictOfReferences = dictOfReferences
        self.__algoDescription = algoDescription
        self.__descriptionGraphicPath = descriptionGraphicPath
        self.__graphicDescription = graphicDescription
        self.__trainingSet = []
        self.__testResults = []
        self.__randomSplitSeed = None
        self.__datafile = datafile
        self.__randomSplitSeed = randomSplitSeed
        self.__classToColor = {}
        self.__dataModelOverview = f"""
        <table>
            <tr>
                <th class="SplitInfoTable">Data:</th>
                <th class="SplitInfoTable">{self.__datafile}</th>
            </tr>
            <tr>
                <th class="SplitInfoTable">Split seed:</th>
                <th class="SplitInfoTable">{self.__randomSplitSeed}</th>
            </tr>
        </table>"""

    def addTrainingSet(self, trainingSet):
        """
        Adds the training set. This is used to visualise the training data used to train the model.

        Parameters
        ----------
        trainingSet : list
            a list containing all training data [['sen','class']]
        """
        self.__trainingSet.append(trainingSet)


    def addTestResults(self, testResults):
        """
        Adds the test results. This is used to visualise the classification performance.

        Parameters
        ----------
        testResults : list
            a list containing all test results [[act,pred]]
        """
        self.__testResults.append(testResults)

    def __createMetrics(self, filepath,MetricsName):

        labels = []

        classesInData =""""""
        dictOfDataSets = {}
        fullDataSet =  {}
        numberOfElements = 0
        boxPlotData = []
        if MetricsName == "Test":
            for metric in self.__testResults:
                labels = list({sample[0] for sample in metric})
            numberOfElements = len(self.__testResults)
            for metric in self.__testResults:
                for sample in metric:
                    if not sample[0] in fullDataSet.keys():
                        fullDataSet[sample[0]] = []

                    if not sample[0] in dictOfDataSets.keys():
                        dictOfDataSets[sample[0]] = 1
                    else:
                        dictOfDataSets[sample[0]] += 1

                for key in labels:
                    if key in dictOfDataSets.keys():
                        fullDataSet[key].append(dictOfDataSets[key])
                    else:
                        fullDataSet[key].append(0)
                dictOfDataSets = {}

        else:
            for metric in self.__trainingSet:
                labels = list({sample[1] for sample in metric})
            numberOfElements = len(self.__trainingSet)
            for metric in self.__trainingSet:
                for sample in metric:
                    if not sample[1] in fullDataSet.keys():
                        fullDataSet[sample[1]] = []

                    if not sample[1] in dictOfDataSets.keys():
                        dictOfDataSets[sample[1]] = 1
                    else:
                        dictOfDataSets[sample[1]] += 1

                for key in labels:
                    if key in dictOfDataSets.keys():
                        fullDataSet[key].append(dictOfDataSets[key])
                    else:
                        fullDataSet[key].append(0)
                dictOfDataSets = {}



        for key in fullDataSet.keys():
            sum = 0

            for num in fullDataSet[key]:
                sum += num
            dictOfDataSets[key] = sum / numberOfElements

        listOfSortedData = []
        for key in dictOfDataSets.keys():

            listOfSortedData.append([dictOfDataSets[key], key])
        listOfSortedData = sorted(
            listOfSortedData, key=lambda x: x[0], reverse=True
        )
        labels = []
        sizes = []
        explode = []
        color = [
            "#F06060",
            "#F2DD64",
            "#F3B562",
            "#F2EBBF",
            "#8CBEB2",
            "#586473",
            "#5C4B51",
            "#4A89AA",
        ]
        colorData = []
        for data, i in zip(
                listOfSortedData, range(len(listOfSortedData))
        ):
            classesInData += f"""<tr>
                        <th class="TrainingDataClasses">{data[1]}</th>
                        <th>{int(data[0])}</th> 
                    </tr>\n"""
            sizes.append(data[0])
            labels.append(data[1])
            explode.append(0)
            if MetricsName == "Training":
                self.__classToColor[data[1]] = color[i % len(color)]
            colorData.append(self.__classToColor[data[1]])
        plt.pie(
            sizes,
            explode=explode,
            labels=labels,
            autopct="%1.1f%%",
            shadow=False,
            startangle=0,
            colors=colorData,
        )
        plt.savefig(
            filepath.replace("file://", '') + f"/PieChart{MetricsName}Data.svg",
            format="svg")
        plt.close()

        for key in labels:
            boxPlotData.append(fullDataSet[key])
        labels.insert(0,"")
        y_pos = np.arange(len(labels))
        for data,i in zip(boxPlotData, range(len(boxPlotData))):
            plt.boxplot(data,positions=[i+1], patch_artist=True,boxprops=dict(facecolor=self.__classToColor[labels[i+1]]))
        plt.xticks(y_pos, labels, rotation=90)
        plt.subplots_adjust(bottom=0.3, top=0.99)
        plt.ylabel("Sampels")
        plt.savefig(filepath.replace("file://",'') + f"/BarChart{MetricsName}Data.png",dpi=600, format="png")
        plt.close()
        labels = labels[1:]

        if MetricsName == "Test":
            folds = list(range(len(fullDataSet[list(fullDataSet.keys())[0]])))
            plt.figure(figsize=(17, 5))
            listOfKeys = list(fullDataSet.keys())
            plt.bar(folds, fullDataSet[listOfKeys[0]], color=self.__classToColor[listOfKeys[0]])
            for i in range(len(listOfKeys)-1):

                bottom = np.array(fullDataSet[listOfKeys[0]])
                for m in range(i):
                    bottom += np.array(fullDataSet[listOfKeys[m+1]])
                plt.bar(folds, fullDataSet[listOfKeys[i+1]], bottom=bottom,color=self.__classToColor[listOfKeys[i+1]])
            plt.legend(listOfKeys)
            plt.savefig(filepath.replace("file://",'') + f"/BarChartOverviewData.svg",format="svg")
            plt.close()
        return (classesInData,labels)


    def createRaport(self, fileName="ModelRaport",htmlDebug = False):
        """
        Created the pdf report of the model

        Parameters
        ----------
        fileName : str
            the name of the pdf raport.
        """

        fileName += ".pdf"


        config = None
        if platform.system() == "Windows":
            config = pdfkit.configuration(wkhtmltopdf="C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe")
            file_path = "file://" + os.path.join(os.getcwd(), "temp").replace('C:','').replace('\\','/')
        else:
            file_path = os.path.join(os.getcwd(), "temp")

        try:
            if platform.system() == "Windows":
                if not os.path.exists(file_path.replace("file://", "C:")):
                    os.mkdir(file_path.replace("file://", "C:"))
            else:
                if not os.path.exists(file_path):
                    os.mkdir(file_path)
        except:
            print("Could not create folder!")

        options = {
            "page-size": "A4",
            "margin-top": "5mm",
            "margin-right": "5mm",
            "margin-bottom": "5mm",
            "margin-left": "5mm",
            "encoding": "UTF-8",
            "enable-local-file-access": True,
        }
        referencesInHTML = ""
        for name in self.__dictOfReferences.keys():
            referencesInHTML += (
                f"""<li><a href="{self.__dictOfReferences[name]}">{name}</a></li>\n"""
            )



        classesInTrainingData, labels = self.__createMetrics(file_path,"Training")
        classesInTestData, labelsTest = self.__createMetrics(file_path,"Test")



        totalConfusionMatrix = {key: {subkey: 0 for subkey in labels} for key in labels}
        listOfIndivitualConfusionsMatrixes = []
        performanceData = {key: {"precision": 0, "recall": 0, "fScore": 0,"N":0} for key in labels}
        totalCorrectClassified = 0
        alltestCases = 0
        for testFrame in self.__testResults:
            individConfMatrix = {key: {subkey: 0 for subkey in labels} for key in labels}
            for sample in testFrame:
                if sample[1] == sample[0]:
                    totalCorrectClassified += 1
                alltestCases += 1
                performanceData[sample[0]]["N"] += 1
                totalConfusionMatrix[sample[1]][sample[0]] += 1
                individConfMatrix[sample[1]][sample[0]] += 1
            listOfIndivitualConfusionsMatrixes.append(individConfMatrix)

        confMatrix = []
        regConfMatrix = []
        numberOfElementsInMatrix = {key : 0 for key in totalConfusionMatrix.keys()}
        for key in totalConfusionMatrix.keys():
            for subkey in totalConfusionMatrix[key].keys():
                numberOfElementsInMatrix[key] += totalConfusionMatrix[subkey][key]
        for key in totalConfusionMatrix.keys():
            line = []
            line2 = []
            for subkey in totalConfusionMatrix[key].keys():
                line.append(totalConfusionMatrix[key][subkey])
                if not numberOfElementsInMatrix[subkey] == 0:
                    line2.append((totalConfusionMatrix[key][subkey]/numberOfElementsInMatrix[subkey])*100)
                else:
                    line2.append(0)
            confMatrix.append(line)
            regConfMatrix.append(line2)
        df_cm = pd.DataFrame(
            confMatrix,
            index=[i + " (pre)" for i in totalConfusionMatrix.keys()],
            columns=[i + " (act)" for i in totalConfusionMatrix.keys()],
        )

        df_refcm =pd.DataFrame(
            regConfMatrix,
            index=[i + " (pre)" for i in totalConfusionMatrix.keys()],
            columns=[i + " (act)" for i in totalConfusionMatrix.keys()],
        )
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.subplots_adjust(bottom=0.3, top=0.99, left=0.2, right=0.99)
        plt.savefig(file_path.replace("file://",'') + "/ConfusionMatrixPerformanceData.png",dpi=600, format="png")
        plt.close()

        plt.figure(figsize=(10, 7))
        sn.heatmap(df_refcm, annot=True)
        plt.subplots_adjust(bottom=0.3, top=0.99, left=0.2, right=0.99)
        plt.savefig(file_path.replace("file://",
                                      '') + "/RegConfusionMatrixPerformanceData.png",
                    dpi=600, format="png")
        plt.close()

        for key in totalConfusionMatrix.keys():
            totalInRow = 0
            totalInColumn = 0
            for subkey in totalConfusionMatrix.keys():
                totalInRow += totalConfusionMatrix[key][subkey]
                totalInColumn += totalConfusionMatrix[subkey][key]
            if totalInRow == 0:
                totalInRow = 1
            if totalInColumn == 0:
                totalInColumn = 1
            performanceData[key]["precision"] = totalConfusionMatrix[key][key] / totalInRow
            performanceData[key]["recall"] = totalConfusionMatrix[key][key] / totalInColumn
            try:
                performanceData[key]["fScore"] = (
                    2
                    * performanceData[key]["precision"]
                    * performanceData[key]["recall"]
                ) / (performanceData[key]["precision"] + performanceData[key]["recall"])
            except:
                performanceData[key]["fScore"] = 0


        fStatByKatAnSample = {key: [] for key in labels}
        for frame in listOfIndivitualConfusionsMatrixes:
            for key in frame.keys():
                totalInRow = 0
                totalInColumn = 0
                for subkey in frame.keys():
                    totalInRow += frame[key][subkey]
                    totalInColumn += frame[subkey][key]
                if totalInRow == 0:
                    totalInRow = 1
                if totalInColumn == 0:
                    totalInColumn = 1
                precision = frame[key][key] / totalInRow
                recall= frame[key][key] /totalInColumn
                try:
                    fStatByKatAnSample[key].append((2*precision * recall) / (precision + recall))
                except:
                    fStatByKatAnSample[key].append(0)

        listOfKeys = [""]
        for key, i in zip(fStatByKatAnSample.keys(),range(len(list(fStatByKatAnSample.keys())))):
            plt.boxplot(fStatByKatAnSample[key], positions=[i + 1], patch_artist=True,boxprops=dict(facecolor=self.__classToColor[key]))
            listOfKeys.append(key)
        plt.xticks(np.arange(len(listOfKeys)), listOfKeys, rotation=90)
        plt.subplots_adjust(bottom=0.3, top=0.99)
        plt.ylabel("F1-Score")
        plt.savefig(file_path.replace("file://",'') + f"/BoxPlotPerformance.png",dpi=600, format="png")
        plt.close()

        plt.figure(figsize=(17, 5))
        for key, i in zip(fStatByKatAnSample.keys(),range(len(list(fStatByKatAnSample.keys())))):
            plt.plot(list(range(len(fStatByKatAnSample[key]))),np.array(fStatByKatAnSample[key])*100,color= self.__classToColor[key])
        plt.ylabel("F1Score (%)")
        plt.xlabel("Splits")
        plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=1)
        plt.grid(which='major', color='#DDDDDD', linewidth=1.2)
        plt.minorticks_on()
        plt.legend(listOfKeys[1:])
        plt.savefig(file_path.replace("file://",'') + f"/PlotFScore.png",dpi=600, format="png")
        plt.close()


        accuracy = totalCorrectClassified/alltestCases
        macroAverage = {"precision": 0,"recall":0, "fScore":0}
        weightedAverage = {"precision": 0, "recall": 0, "fScore": 0}
        for scoreType in list(performanceData[list(performanceData.keys())[0]].keys())[:-1]:
            sumMacroAverage = 0
            divMacroAverage = 0
            sumWeightedAverage = 0
            divWeightedAverage = 0
            for key in performanceData.keys():
                sumMacroAverage += performanceData[key][scoreType]
                divMacroAverage += 1
                sumWeightedAverage += performanceData[key][scoreType] * performanceData[key]["N"]
                divWeightedAverage += performanceData[key]["N"]
            macroAverage[scoreType] = sumMacroAverage/divMacroAverage
            weightedAverage[scoreType] = sumWeightedAverage/divWeightedAverage

        classificationPerformanceTable = """"""
        for key in performanceData.keys():
            classificationPerformanceTable += f"""<tr>
                <th class="TrainingDataClasses">{key}</th>
                <th class="ImgCell">{performanceData[key]['precision']*100:.2f}%</th>
                <th class="ImgCell">{performanceData[key]['recall']*100:.2f}%</th>
                <th class="ImgCell">{performanceData[key]['fScore']*100:.2f}%</th>
                </tr>\n"""

        classificationPerformanceTable += f"""<tr>
                <th class="HorizontalBar TrainingDataClasses Bold">Accuracy</th>
                <th class="HorizontalBar ImgCell"></th>
                <th class="HorizontalBar ImgCell"></th>
                <th class="HorizontalBar ImgCell">{accuracy*100:.2f}%</th>
                </tr>\n"""

        classificationPerformanceTable += f"""<tr>
                <th class="TrainingDataClasses Bold">Macro Average</th>
                <th class="ImgCell">{macroAverage['precision']*100:.2f}%</th>
                <th class="ImgCell">{macroAverage['recall']*100:.2f}%</th>
                <th class="ImgCell">{macroAverage['fScore']*100:.2f}%</th>
                </tr>\n"""

        classificationPerformanceTable += f"""<tr>
                <th class="TrainingDataClasses Bold">Weighted Average</th>
                <th class="ImgCell">{weightedAverage['precision']*100:.2f}%</th>
                <th class="ImgCell">{weightedAverage['recall']*100:.2f}%</th>
                <th class="ImgCell">{weightedAverage['fScore']*100:.2f}%</th>
                </tr>\n"""





        htmlTemplate = (
            """<html>
    <head>
        <style type="text/css">
            .break-before {
                page-break-before: always;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
                    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
                    sans-serif;
                font-size: 80%;
            }
    
            .Header {
                display: -webkit-box;
                position: relative;
                top: -50px;
            }
    
            .TitleWithText {
                padding-left: 20px;
                font-weight: bold;
                font-size: larger;
            }
    
            .Logo {
                position: relative;
                height: 60px;
                padding-left: 40px;
                width: 200px;
                top: -16px;
                right: -770px;
            }
    
            svg {
                height: 100%;
            }
    
            .Overview {
                padding-left: 20px;
                display: -webkit-box;
            }
    
            h4 {
                padding-bottom: 0px;
                margin-bottom: 0px;
            }
    
            h2 {
                padding-left: 10px;
            }
    
            .MiddleOverview {
                padding-left: 40px;
            }
    
            .AlgoDescription {
                display: block;
                width: 500px;
                font-weight: lighter;
                font-size: 100%;
            }
    
            .OverviewImg {
                height: 200px;
                
            }
    
            .RightOverview {
                padding-left: 20px;
            }
    
            .TrainingDataset {
                padding-left: 20px;
            }
    
            .TrainingDataTable {
                margin-top: 20px;
                font-family: arial, sans-serif;
                width: 300px;
                border-collapse: collapse;
                margin-left: 40px;
                font-weight: lighter;
                
            }
            
            .TrainingDataClasses
            {
                text-align: left;
            }
    
            th {
                font-size: small;
                font-weight: normal;
                text-align: center;
                padding-left: 3px;
            }
    
            td,
            th {
                border: 1px solid #dddddd;
                border-bottom: transparent;
                border-top: transparent;
                border-left: transparent;
                font-size: 15 px;
    
            }
            
            .HorizontalBar{
                border-top-width: 3px;
                border-top-style: solid;
                border-top-color: #dadada;
            }
            
            .Bold{
                font-weight: bold;
            }
            
            
            .SplitInfoTable {
                border-right: transparent;
                padding-right: 10px;
                text-align: left;
            }
    
            .tableHeader {
                font-size: 14px;
                font-weight: bolder;
            }
    
            .PiChartTrainingData {
                width: 250px;
                height: 250px;
                padding-left: 0px;
            }
    
    
            .TrainingDataView {
                display: -webkit-box;
                padding-bottom: 20px;
            }
     
            .BarChartTrainingData {
                width: 250px;
                height: 250px;
                padding-left: 80px;
            }
    
            .ClassificationPerformanceTable {
                margin-top: 20px;
                font-family: arial, sans-serif;
                border-collapse: collapse;
                margin-left: 40px;
                width: 500px;
            }
    
            .ImgCell {
                padding-top: 10px;
                height: 20px;
            }
    
            .PerformancePlots {
                display: -webkit-box;
                padding-top: 50px;
            }
            
            .OverviewClassificationPerformance {
                display: -webkit-box;
                padding-top: 50px;
            }
    
            .ROC,
            .ConfusionMatrix {
                height: 350px;
                align-items: flex-start;
            }
            
            .infoLabel{
                font-size: 8px;
                font-weight: 300;
                padding: 2px;
                margin: 2px;
            }
    
            .svgImage {
                height: 100%;
            } 
    
            .VerticalSeprator {
                width: 200px;
                height: 80%;
            }
    
            .h4PerformacePlots {
                padding-bottom: 30px;
            }
            
            .StackedGroupedBarChartDataSet{
                height: 300px;
            }
            
            .F1ScoreBySplit
            {
                padding-top: 50px;
                height:300px;
            }
        </style>
    </head>
    <h1>Model Performance</h1>
    <div class="Logo">
        <svg id="Logo" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2504.67 1216.48">
            <defs>
                <style>
                    .cls-1 {
                        fill: #191919;
                    }
    
                    .cls-2 {
                        fill: #d72864;
                    }
    
                    .cls-3 {
                        fill: #8c195f;
                    }
                </style>
            </defs>
            <path class="cls-1"
                d="M2342.48,880.61c-108.87,0-194.15,84.09-194.15,191.46,0,112.19,83.47,196.8,194.15,196.8s194.14-84.61,194.14-196.8C2536.62,964.7,2451.35,880.61,2342.48,880.61Zm0,330.34c-71.88,0-128.18-61-128.18-138.88,0-74.9,56.3-133.55,128.18-133.55s128.17,58.65,128.17,133.55C2470.65,1150,2414.35,1211,2342.48,1211Z"
                transform="translate(-667.67 -471.76)" />
            <path class="cls-1"
                d="M2749,1047.63c-38.48-15.92-71.73-29.67-71.73-60.3,0-27.52,25.83-48.28,60.08-48.28,41.49,0,74.13,24.14,83.4,31.74l27.5-51.59c-7.45-6.93-45.07-38.59-110.36-38.59-71.24,0-125,46.33-125,107.79,0,64.53,56.42,88.86,106.2,110.32,39.41,17,73.47,31.66,73.47,62.91,0,33.71-29.09,48.8-57.92,48.8-47.59,0-85-30.43-94.52-38.87l-34.47,47.73c8.57,8.55,53.66,49.58,127.92,49.58,72.31,0,122.81-44.76,122.81-108.86C2856.36,1092.06,2799.3,1068.44,2749,1047.63Z"
                transform="translate(-667.67 -471.76)" />
            <polygon class="cls-1"
                points="2504.67 415.28 2261.18 415.28 2261.18 467.83 2352.9 467.83 2352.9 790.7 2412.43 790.7 2412.43 467.83 2504.67 467.83 2504.67 415.28" />
            <path class="cls-1"
                d="M1707.48,651.81c-110.27-111.13-263-180-431.58-180-335.38,0-608.23,272.86-608.23,608.25s272.85,608.23,608.23,608.23c167.69,0,319.75-68.21,429.88-178.35S2008,1247.7,2008,1080C2008,913.17,1816.62,761.81,1707.48,651.81Zm-30.61,773.34c-88.11,88.11-209.75,142.68-343.91,142.68-15.06,0-29.94-.79-44.67-2.13-7.5-.69-14.94-1.57-22.34-2.59-14.7,2-29.59,3.47-44.67,4.16-7.41.33-14.85.56-22.35.56-268.31,0-486.59-218.28-486.59-486.58s218.28-486.6,486.59-486.6c7.48,0,14.89.23,22.29.57,15.09.68,30,2.12,44.71,4.16,7.39-1,14.83-1.91,22.33-2.59,14.73-1.35,29.63-2.14,44.7-2.14,134.84,0,257,55.13,345.26,144,87.32,88,240.45,209.09,240.45,342.56C1918.67,1215.4,1765,1337.05,1676.87,1425.15Z"
                transform="translate(-667.67 -471.76)" />
            <path class="cls-2"
                d="M846.38,1081.25c0-245.56,177.07-444.08,419.55-481.87h0C966.81,606.57,757,820.45,757,1081.25s251.49,488.93,508.91,481.86h0C1029.25,1530.34,846.38,1326.81,846.38,1081.25Z"
                transform="translate(-667.67 -471.76)" />
            <path class="cls-3"
                d="M1587.53,1425.15c-89.74,89.75-205.23,127.31-321.58,138,7.4,1,14.84,1.9,22.34,2.59,14.73,1.34,29.61,2.13,44.67,2.13,134.15,0,255.8-54.57,343.91-142.68s241.8-209.75,241.8-343.9c0-133.47-153.13-254.56-240.45-342.56-88.21-88.91-210.42-144-345.26-144-15.08,0-30,.79-44.7,2.14-7.5.68-14.94,1.56-22.33,2.59,117,10.7,244.59,60.33,323,139.31,87.31,88,240.44,209.09,240.44,342.56C1829.33,1215.4,1675.64,1337,1587.53,1425.15Z"
                transform="translate(-667.67 -471.76)" />
            <path class="cls-3"
                d="M757,1081.25c0-260.8,209.69-474.68,508.89-481.87-14.72-2-29.62-3.48-44.71-4.16-7.4-.34-14.82-.57-22.29-.57-268.31,0-486.59,218.29-486.59,486.6s218.28,486.58,486.59,486.58c7.5,0,14.94-.23,22.35-.56,15.08-.69,30-2.13,44.67-4.16C1008.9,1570.18,757,1342.06,757,1081.25Z"
                transform="translate(-667.67 -471.76)" />
            <path class="cls-2"
                d="M1542.84,1425.15a485.68,485.68,0,0,1-276.89,138c116.35-10.65,231.84-48.21,321.58-138,88.11-88.11,241.8-209.75,241.8-343.9,0-133.47-153.13-254.56-240.44-342.56-78.37-79-205.92-128.61-323-139.31a485.62,485.62,0,0,1,278.26,139.31c87.32,88,240.45,209.09,240.45,342.56C1784.64,1215.4,1631,1337.05,1542.84,1425.15Z"
                transform="translate(-667.67 -471.76)" />
        </svg>
    </div>
    <div class="Header">
"""
            + f"""<div class="ModelName">
            <label class="TitleWithText">Model Name:</label>
            <label>{self.__modelName}</label>
        </div>
        <div class="TestDate">
            <label class="TitleWithText">Test Date:</label>
            <label>{self.__date}</label>
        </div>
        <div class="Creator">
            <label class="TitleWithText">Creator:</label>
            <label>{self.__creatorName}</label>
        </div>
    
    </div>
    
    <h2>Overview</h2>
    <div class="Overview">
        <div class="LeftOverview">
            <h4>ML Principle:</h4>
            <label>{self.__MLPrinciple}</label>
    
            <h4>References:</h4>
            <ul>
                {referencesInHTML}
            </ul>
    
        </div>
    
        <div class="MiddleOverview">
            <h4>Algorithm Description:</h4>
            <label class="AlgoDescription">{self.__algoDescription}</label>
        </div>
    
        <div class="RightOverview">
            <img class="OverviewImg" src="{self.__descriptionGraphicPath}" alt="Overview Image">
            <h6>{self.__graphicDescription}</h6>
        </div>
    </div>
    <hr>
    </hr>
    
    <h2>Metrics</h2>
    <label>{self.__dataModelOverview}</label>
    <div class="TrainingDataset">
        <h4>Training Dataset</h4>
        <label class="infoLabel">(average)</label>
        <div class="TrainingDataView">
            <table class="TrainingDataTable">
                <tr>
                    <th class="tableHeader TrainingDataClasses">Classes</th>
                    <th class="tableHeader">Number of samples</th>
                </tr>
                {classesInTrainingData} 
            </table>
    
            <div class="PiChartTrainingData">
                <img class="svgImage" src="{file_path}/PieChartTrainingData.svg" alt="PlotSample">
                <label class="infoLabel">Average distribution of the samples</label>
            </div>
    
            <div class="BarChartTrainingData">
                <img class="svgImage" src="{file_path}/BarChartTrainingData.png" alt="PlotSample">
                <label class="infoLabel">Distribution of the samples contained in each test split</label>
            </div>
        </div>
    </div>
    
    <div class="TrainingDataset">
        <h4>Test Dataset</h4>
        <label class="infoLabel">(average)</label>
        <div class="TrainingDataView">
            <table class="TrainingDataTable">
                <tr>
                    <th class="tableHeader TrainingDataClasses">Classes</th>
                    <th class="tableHeader">Number of samples</th>
                </tr>
                {classesInTestData} 
            </table>
    
            <div class="PiChartTrainingData">
                <img class="svgImage" src="{file_path}/PieChartTestData.svg" alt="PlotSample">
                <label class="infoLabel">Average distribution of the samples</label>
            </div>
    
            <div class="BarChartTrainingData">
                <img class="svgImage" src="{file_path}/BarChartTestData.png" alt="PlotSample">
                <label class="infoLabel">Distribution of the samples contained in each test split</label>
            </div>
        </div>
    </div>
    
    <div class="StackedGroupedBarChartDataSet">
        <img class="svgImage" src="{file_path}/BarChartOverviewData.svg" alt="PlotSample">
        <label class="infoLabel">Detailed training split composition</label>
    </div>

        <h2 class="break-before">Classification Performance</h2>
        <div class="ClassificationPerformance">
            <div class="OverviewClassificationPerformance">
                <div>
                    <table class="ClassificationPerformanceTable">
                        <tr>
                            <th class="tableHeader TrainingDataClasses">Classes</th>
                            <th class="tableHeader">Precision</th>
                            <th class="tableHeader">Recall</th>
                            <th class="tableHeader">F1 Score</th>
                        </tr>
                            {classificationPerformanceTable}
                    </table>
                </div>
                <div class="BarChartTrainingData">
                    <img class="svgImage" src="{file_path}/BoxPlotPerformance.png" alt="PlotSample">
                    <label class="infoLabel">Distribution of the F1-Score</label>
                </div>
            </div>
            <div class="PerformancePlots">
                <div class="ConfusionMatrix">
                    <h4 class="h4PerformacePlots">ConfusionMatrix:</h4>
                    <img class=" svgImage" src="{file_path}/ConfusionMatrixPerformanceData.png" alt="PlotSample">
                </div>
                <div class="ConfusionMatrix">
                    <h4 class="h4PerformacePlots">Normalised ConfusionMatrix:</h4>
                    <img class=" svgImage" src="{file_path}/RegConfusionMatrixPerformanceData.png" alt="PlotSample">
                </div>
            </div>
            <div class="F1ScoreBySplit">
                <h4>F1 Socre by split:</h4>
                <img class="svgImage" src="{file_path}/PlotFScore.png" alt="PlotSample">
                <label class="infoLabel">F1-Score per split</label>
            </div>
        </div>
    </div>
    </html>"""
        )

        if htmlDebug:
            print(htmlTemplate)
            with open("debugFile.html",'w') as out:
                out.write(htmlTemplate)

            pdfkit.from_file("debugFile.html",fileName, options=options, configuration=config)
        else:
            pdfkit.from_string(
                htmlTemplate, fileName, options=options, configuration=config
            )
        self.__trainingSet = None
        self.__testResults = None
        print(f"File created ->{file_path.replace('/temp','/')+fileName}")