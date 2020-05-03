from repository import *
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def getTrainData(inputs):
    return inputs[:int(0.8 * (len(inputs)))]


def getTestData(inputs):
    return inputs[int(0.8 * (len(inputs))):]


def normalisation(trainData, testData):
    scaler = StandardScaler()
    if not isinstance(trainData[0], list):
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]

        scaler.fit(trainData)
        normalisedTrainData = scaler.transform(trainData)
        normalisedTestData = scaler.transform(testData)

        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        scaler.fit(trainData)
        normalisedTrainData = scaler.transform(trainData)
        normalisedTestData = scaler.transform(testData)
    return normalisedTrainData, normalisedTestData


# cu tool
def run_tool():
    # impartire
    inputs_setosa, inputs_versicolor, inputs_virginica = load_data()

    setosa_antrenament = getTrainData(inputs_setosa)
    versicolor_antrenament = getTrainData(inputs_versicolor)
    virginica_antrenament = getTrainData(inputs_virginica)

    setosa_test = getTestData(inputs_setosa)
    versicolor_test = getTestData(inputs_versicolor)
    virginica_test = getTestData(inputs_virginica)

    # normalizare
    # antrenament_normalised_setosa, test_normalised_setosa = normalisation(setosa_antrenament, setosa_test)
    # antrenament_normalised_versicolor, test_normalised_versicolor = normalisation(versicolor_antrenament, versicolor_test)
    # antrenament_normalised_virginica, test_normalised_virginica = normalisation(virginica_antrenament, virginica_test)
    #
    # inputs = []
    # outputs = []
    # for i in antrenament_normalised_setosa:
    #     inputs.append(i)
    #     outputs.append(0)
    # for i in antrenament_normalised_versicolor:
    #     inputs.append(i)
    #     outputs.append(1)
    # for i in antrenament_normalised_virginica:
    #     inputs.append(i)
    #     outputs.append(2)

    # normalizare

    inputs_antrenament = []
    inputs_test = []
    outputs_antrenament = []
    outputs_test = []
    for i in range(len(setosa_antrenament)):
        inputs_antrenament.append(setosa_antrenament[i])
        outputs_antrenament.append(0)
        inputs_antrenament.append(versicolor_antrenament[i])
        outputs_antrenament.append(1)
        inputs_antrenament.append(virginica_antrenament[i])
        outputs_antrenament.append(2)

    for i in range(len(setosa_test)):
        inputs_test.append(setosa_test[i])
        outputs_test.append(0)
        inputs_test.append(versicolor_test[i])
        outputs_test.append(1)
        inputs_test.append(virginica_test[i])
        outputs_test.append(2)

    antrenament_norm, test_norm = normalisation(inputs_antrenament, inputs_test)

    # antrenament
    classifier = linear_model.LogisticRegression(multi_class='ovr')
    classifier.fit(antrenament_norm, outputs_antrenament)
    print(classifier.intercept_)
    print(classifier.coef_)

    # test
    computedTestOutputs = classifier.predict(test_norm)
    error = 1 - accuracy_score(outputs_test, computedTestOutputs)
    print("classification error (tool): ", error)

run_tool()
