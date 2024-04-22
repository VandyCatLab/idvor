import glob
import pandas as pd
import os
import itertools
import numpy as np
import scipy.stats as stats
import json
import glob
import tensorflow as tf
import gc
import time


def compile_correspondence(path, models_path):
    """
    Return dataframe of the correspondence data from path.
    """
    files = glob.glob(os.path.join(path, "*.csv"))

    # Compile data
    df = pd.DataFrame()
    for file in files:
        tmp = pd.read_csv(file, index_col=0)
        # Remove rows that aren't the correct models
        tmp = tmp.loc[tmp.model2.apply(lambda x: "w" in x and "s" in x)]
        # Save over old file
        tmp.to_csv(file)
        layerCol = list(range(11)) * int(len(tmp) / 11)
        tmp["layer"] = layerCol
        df = pd.concat((df, tmp))

    df["layer"] = df["layer"].astype(int)

    # Figure out which model combination is missing
    models = glob.glob(os.path.join(models_path, "*"))
    models = [model.split("/")[-1].split(".")[0] for model in models]
    missing = []
    for pair in itertools.combinations(models, 2):
        test = df[["model1", "model2"]] == pair
        test = test.all(axis=1)
        if not test.any():
            # Test reversed direction
            test = df[["model2", "model1"]] == pair
            test = test.all(axis=1)
            if not test.any():
                missing += [pair]
                continue

    return df, pd.DataFrame(missing, columns=["model1", "model2"])


def correspondence_missing_optimizer(missing):
    """
    Given the missing combinations, return an optimized list to load as few
    models as possible to complete the dataset.
    """
    data = missing.copy()
    modelPairs = {}
    models = list(np.unique(np.concatenate((data["model1"], data["model2"]))))

    while len(models) > 0:
        modelCounts = [[]] * len(models)
        for i, model in enumerate(models):
            modelCounts[i] = list(data["model2"].loc[data["model1"] == model])
            modelCounts[i] += list(data["model1"].loc[data["model2"] == model])

        modelN = np.array([len(count) for count in modelCounts])
        target = models.pop(np.argmax(modelN))
        modelPairs[target] = list(data["model1"].loc[data["model2"] == target]) + list(
            data["model2"].loc[data["model1"] == target]
        )
        data = data.loc[(data["model1"] != target) & (data["model2"] != target)]

    return modelPairs


def compile_augment(path, augment, layer):
    """
    Return the data of certain type of baseline augment from path at a given
    layer.
    """
    files = glob.glob(os.path.join(path, f"*l{layer}-{augment}*.csv"))
    df = pd.DataFrame()
    for file in files:
        tmp = pd.read_csv(file)
        tmp["layer"] = layer
        df = pd.concat((df, tmp))

    if augment == "color":
        df["version"] = (df["version"] - 25) * (3 / 50)
    elif "translate" in augment:
        metricNames = [name for name in df.columns if not name in ["version", "layer"]]
        # Remove directions from metricNames
        metricNames = [name.split("-")[0] for name in metricNames]
        # Keep only unique names
        metricNames = list(np.unique(metricNames))

        # For each metric, average all directions
        for metric in metricNames:
            # Get columns with the metric
            cols = [name for name in df.columns if metric in name]
            # Get average of all directions
            df[metric] = df[cols].mean(axis=1)

    return df


def compile_baseline_dict(path, tests, layers, metrics):
    """
    Return a dictionary of the results from the list baseline tests, grouped
    by test then layer then metrics for d3 plotting.
    """
    data = {test: {} for test in tests}
    for testKey in data.keys():
        layerData = {f"layer{layer}": {} for layer in layers}
        metricData = {metricKey: [] for metricKey in metrics}
        if testKey == "dropout":  # This is dropout
            df = compile_dropout(path)
            versions = df["dropRate"].unique()
            for layerKey in layerData.keys():
                for metric in metrics:
                    for version in versions:
                        tmpData = df[metric].loc[
                            (df["dropRate"] == version)
                            & (df["layer"] == int(layerKey[5:]))
                        ]
                        mean = np.mean(tmpData)
                        ciLow, ciHigh = (
                            stats.t.interval(
                                alpha=0.95,
                                df=len(tmpData) - 1,
                                loc=mean,
                                scale=stats.sem(tmpData, nan_policy="omit"),
                            )
                            if stats.sem(tmpData, nan_policy="omit") > 0
                            else (mean, mean)  # No variance case catch
                        )

                        # Save data of a metric
                        metricData[metric] += [
                            {
                                "test": testKey,
                                "layer": layerKey,
                                "version": version,
                                "metric": metric,
                                "mean": mean,
                                "ciLow": ciLow,
                                "ciHigh": ciHigh,
                            }
                        ]
                # Save data of a layer
                layerData[layerKey] = metricData
            # Save data of a test
            data[testKey] = layerData

        else:  # Probably augmentation
            for layerKey in layerData.keys():
                df = compile_augment(path, testKey, int(layerKey[5:]))
                versions = df["version"].unique()

                for metric in metrics:
                    if testKey == "translate":  # Translation test
                        directions = ["left", "right", "up", "down"]
                        dirKeys = [f"{metric}-{direction}" for direction in directions]
                        df[metric] = df[dirKeys].mean(1)
                    for version in versions:
                        tmpData = df[metric].loc[df["version"] == version]
                        mean = np.mean(tmpData)
                        ciLow, ciHigh = (
                            stats.t.interval(
                                alpha=0.95,
                                df=len(tmpData) - 1,
                                loc=mean,
                                scale=stats.sem(tmpData, nan_policy="omit"),
                            )
                            if stats.sem(tmpData, nan_policy="omit") > 0
                            else (mean, mean)  # No variance case catch
                        )

                        # Save data of a metric
                        metricData[metric] += [
                            {
                                "test": testKey,
                                "layer": layerKey,
                                "version": version,
                                "metric": metric,
                                "mean": mean,
                                "ciLow": ciLow,
                                "ciHigh": ciHigh,
                            }
                        ]

                        if testKey == "translate":  # Add directions for translation
                            for i, dirCol in enumerate(dirKeys):
                                metricData[metric][-1][directions[i]] = np.mean(
                                    df[dirCol].loc[df["version"] == version]
                                )

                    # Save data of a layer
                    layerData[layerKey] = metricData
                # Save data of a test
                data[testKey] = layerData

    return data


def compile_dropout(path, layers):
    """
    Return the data from the baseline dropout representations from a path.
    """
    layerName = ",".join([str(layer) for layer in layers])
    files = glob.glob(os.path.join(path, f"*dropout-{layerName}.csv"))

    df = pd.DataFrame()
    for file in files:
        tmp = pd.read_csv(file)
        df = pd.concat((df, tmp))

    df = df.rename(columns={"dropRate": "version"})

    return df


def compile_augment_accuracy(path, aug):
    """
    Return the data from the augment accuracy from a path.
    """
    files = glob.glob(os.path.join(path, f"*-acc-{aug}.csv"))

    df = pd.DataFrame()
    for file in files:
        tmp = pd.read_csv(file)
        df = pd.concat((df, tmp))

    if augment == "color":
        df["version"] = (df["version"] - 25) * (3 / 50)

    return df


def hub_rep_completion(repDir, hubInfo):
    """
    Return a list of missing Tensorflow Hub representations in repDir given
    hubInfo.
    """
    # Get list of rep files
    repFiles = glob.glob(os.path.join(repDir, "*.npy"))
    repFiles = [name.replace(repDir + "/", "") for name in repFiles]
    with open(hubInfo, "r") as f:
        hubModels = json.loads(f.read())

    # Fix up model names
    hubModels = list(hubModels.keys())
    hubModels = [name.replace("/", "-") for name in hubModels]

    missing = []
    for name in hubModels:
        if not name + "-Reps.npy" in repFiles:
            print(f"Missing representations from model: {name}")
            missing += [name]

    return missing


def limit_hub_rep_feature_size(repDir, featureLimit):
    """
    Return a list of hub model representations in repDir that have less than or
    equal to featureLimit features.
    """
    # Get a list of rep files
    repFiles = glob.glob(os.path.join(repDir, "*.npy"))

    # Loop through rep files
    keptReps = []
    for file in repFiles:
        # Load rep
        rep = np.load(file)
        # Check if it has less than or equal to featureLimit features
        if rep.shape[-1] <= featureLimit:
            print(f"{file} has {rep.shape[-1]} features, keeping!")
            keptReps += [file]
        else:
            print(f"{file} has {rep.shape[-1]} features, too many!")

    print(repDir)
    return keptReps


def hubSims_completion_check(simDir, hubInfo, featureLimit=None):
    """
    Return a list of missing Tensorflow Hub similarities in simDir given
    hubInfo. If feature limit is set, check that the number of features is
    valid.
    """
    # Load hub model info
    with open(hubInfo, "r") as f:
        hubModels = json.loads(f.read())

    # Create combination of hub models
    hubModels = list(hubModels.keys())

    # Loop through models and look for the corresponding simliarity file
    missing = []
    for model1 in hubModels:
        # Fix model name
        model1 = model1.replace("/", "-")

        if not model1 + ".csv" in os.listdir(simDir):
            # Check if this model that was skipped because it was too big
            if featureLimit is not None:
                rep = np.load(os.path.join(simDir, "..", model1 + "-Reps.npy"))
                if rep.shape[-1] > featureLimit:
                    print(f"{model1} has {rep.shape[-1]} features, skipped!")
                    continue
            print(f"Missing similarity from model: {model1}")
            missing += [model1]

    return missing


def compile_hub_sims(simDir):
    """
    Return a compiled dataframe of similarity data from simDir.
    """
    # Get list of sim files
    simFiles = glob.glob(os.path.join(simDir, "*.csv"))

    # Loop through sim files
    df = pd.DataFrame()
    for file in simFiles:
        tmp = pd.read_csv(file, index_col=0)
        df = pd.concat((df, tmp))

    return df


def compile_noise_sim(simDir):
    """
    Return a compiled dataframe of similarity data from simDir.
    """
    # Get list of sim files
    simFiles = glob.glob(os.path.join(simDir, "noise*.csv"))

    # Loop through sim files
    df = pd.DataFrame()
    for file in simFiles:
        tmp = pd.read_csv(file, index_col=0)
        df = pd.concat((df, tmp))

    return df


def compile_training_traj(trajDir, pattern):
    """
    Return the compiled model training trajectories from the logs in trajDir
    matching the pattern.
    """
    # Get list of log files
    logFiles = glob.glob(os.path.join(trajDir, pattern))

    # Loop through log files
    df = pd.DataFrame(columns=["model", "epoch", "valAcc", "log"])
    for file in logFiles:
        # Open log file
        with open(file, "r") as f:
            # Read log file
            log = f.readlines()
        # Filter lines with validation accuracy
        log = [line for line in log if "val_accuracy" in line]

        # Just grab validation accuracy
        valAcc = [float(line.split(":")[-1].replace("\n", "").strip()) for line in log]

        # Get model name
        model = file.split("/")[-1].split(".")[0]

        # Get epochs
        epochs = range(len(valAcc))

        # Add to dataframe
        df = df.append(
            pd.DataFrame(
                {
                    "model": model,
                    "epoch": epochs,
                    "valAcc": valAcc,
                    "log": file.split("/")[-1],
                }
            )
        )

    return df


def manual_gpu_batch(model, images, batch_size=32):
    # Start timer
    start = time.time()

    # Manual batch
    reps = []
    for i in range(0, images.shape[0], batch_size):
        with tf.device("/gpu"):
            reps += [model(images[i : i + batch_size], training=False)]

    with tf.device("/cpu"):
        # Check if it is multi output
        if isinstance(reps[0], list):
            # Check how many outputs
            nOuts = len(reps[0])
            tmp = [0] * nOuts
            for i in range(nOuts):
                tmp[i] = tf.concat([rep[i] for rep in reps], axis=0)
            reps = tmp
        else:
            reps = tf.concat(reps, axis=0)

    # End timer
    end = time.time()

    print(f"Batch finished in {end - start} seconds")
    return reps


def clear_model():
    # Check if model variable exists
    if "model" in globals():
        del globals()["model"]
        gc.collect()
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    # List globs that match
    modelName = "AlexNet"
    files = glob.glob(f"../outputs/masterOutput/baseline/{modelName}*-random.csv")
    df = pd.DataFrame()
    for file in files:
        tmp = pd.read_csv(file, index_col=0)
        # Figure out what layer from file name
        layer = int(file.split(modelName)[-1].split("l")[0])
        # Add layer column
        tmp["layer"] = layer
        df = pd.concat((df, tmp))

    # Sort by layer
    df = df.sort_values(by="layer")

    # Save df
    df.to_csv(f"../outputs/masterOutput/baseline/compiled/{modelName}-random.csv")
    print(files)

    # layers = [2, 6, 10]
    # path = "../outputs/masterOutput/baseline/catDiff_max1-100"
    # df = compile_dropout(path, layers)
    # df.to_csv(
    #     f"../outputs/masterOutput/baseline/compiled/dropout-catDiff_max1-100.csv"
    # )

    # path = "../outputs/masterOutput/baseline/catDiff_max1-100/"
    # augments = ["translate", "reflect", "zoom"]

    # for augment in augments:
    #     df = pd.DataFrame()
    #     for layer in layers:
    #         tmp = compile_augment(path, augment, layer)
    #         df = pd.concat((df, tmp))
    #     df.to_csv(
    #         f"../outputs/masterOutput/baseline/compiled/{augment}-catDiff_max1-100.csv"
    #     )

    # path = "../outputs/masterOutput/baseline/kriegset"
    # df = compile_dropout(path, layers)
    # df.to_csv(
    #     f"../outputs/masterOutput/baseline/compiled/dropout-kriegset.csv"
    # )

    # path = "../outputs/masterOutput/baseline/kriegset"
    # augments = ["translate", "reflect", "noise", "color", "zoom"]

    # for augment in augments:
    #     df = pd.DataFrame()
    #     for layer in layers:
    #         tmp = compile_augment(path, augment, layer)
    #         df = pd.concat((df, tmp))
    #     df.to_csv(
    #         f"../outputs/masterOutput/baseline/compiled/{augment}-kriegset.csv"
    #     )

    # path = "../outputs/masterOutput/correspondence/"
    # models_path = "../outputs/masterOutput/models/"
    # results, missing = compile_correspondence(path, models_path)
    # missing = correspondence_missing_optimizer(missing)
    # results.to_csv(f"../outputs/masterOutput/correspondence.csv")
    # missing

    # path = "../outputs/masterOutput/correspondence/cka2/"
    # models_path = "../outputs/masterOutput/models/"
    # resultsCKA2, missing = compile_correspondence(path, models_path)
    # missing = correspondence_missing_optimizer(missing)
    # # Replace cka column with cka2 results
    # results.drop(columns="cka", inplace=True)
    # results[["cka"]] = resultsCKA2.cka
    # results.to_csv(f"../outputs/masterOutput/correspondence.csv")
    # missing

    # path = "../outputs/masterOutput/correspondence/pwcca/"
    # models_path = "../outputs/masterOutput/models/"
    # results, missing = compile_correspondence(path, models_path)
    # missing = correspondence_missing_optimizer(missing)
    # results.to_csv(f"../outputs/masterOutput/pwccaCorrespondence.csv")
    # missing

    # modelType = "itemDiff_max10"
    # path = f"../outputs/masterOutput/baseline/itemDiff_max10"
    # augments = ["random"]
    # layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # for augment in augments:
    #     df = pd.DataFrame()
    #     for layer in layers:
    #         tmp = compile_augment(path, augment, layer)
    #         # Add layer column
    #         tmp["layer"] = layer
    #         df = pd.concat((df, tmp))
    #     df.to_csv(
    #         f"../outputs/masterOutput/baseline/compiled/{augment}-{modelType}.csv"
    #     )

    # path = "./outputs/masterOutput/baseline"
    # layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # df = pd.DataFrame()
    # for layer in layers:
    #     tmp = compile_augment(path, "random", layer)
    #     # Add layer column
    #     tmp["layer"] = layer
    #     df = pd.concat((df, tmp))
    # df.to_csv(
    #     f"./outputs/masterOutput/baseline/compiled/random-seedDiff.csv"
    # )
    # path = "../outputs/masterOutput/baseline/cinic/"
    # augments = ["color", "translate", "zoom", "reflect", "noise", "drop"]
    # for augment in augments:
    #     df = compile_augment_accuracy(path, augment)
    #     df.to_csv(
    #         f"../outputs/masterOutput/baseline/compiled/{augment}Acc-cinic.csv"
    #     )

    # missing = hub_rep_completion(
    #     "../outputs/masterOutput/hubReps", "./hubModels.json"
    # )
    # for file in missing:
    #     print(file)

    # kept = limit_hub_rep_feature_size("../outputs/masterOutput/hubReps", 2048)

    # missing = hubSims_completion_check(
    #     "../outputs/masterOutput/hubReps/hubSims", "./hubModels.json", 2048
    # )

    # Compile hub sims
    # df = compile_hub_sims("../outputs/masterOutput/hubReps/hubSims/novset")
    # df.to_csv("../outputs/masterOutput/hubSimsNovset.csv")

    # Compile noise sims
    # df = compile_noise_sim("../outputs/masterOutput/simulation")
    # df.to_csv("../outputs/masterOutput/bigNoiseSim.csv")

    # Compile trajectory
    # df = compile_training_traj(
    #     "../logs/master/training", "train_itemDiff_max3_*"
    # )
    # df.to_csv(
    #     "../outputs/masterOutput/trainingTraj_itemDiff_max3.csv", index=False
    # )

    # Convert similarity matrix numpy format to csv
    # simFuns = ["eucRsa", "cka"]
    # layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # modelSeeds = pd.read_csv("../outputs/masterOutput/modelSeeds.csv")
    # for layer in layers:
    #     # load similarity matrices into dictionary
    #     sims = {}
    #     for simFun in simFuns:
    #         sims[simFun] = np.load(
    #             f"../outputs/masterOutput/similarities/simMat_l{layer}_{simFun}.npy"
    #         )

    #     df = pd.DataFrame()
    #     # Read only triangle
    #     for i in range(sims[simFuns[0]].shape[0]):
    #         for j in range(0, i):
    #             simDict = {fun: [sim[i, j]] for fun, sim in sims.items()}
    #             df = df.append(
    #                 pd.DataFrame(
    #                     {
    #                         "model1": f"w{modelSeeds.weight[i]}s{modelSeeds.shuffle[i]}",
    #                         "model2": f"w{modelSeeds.weight[j]}s{modelSeeds.shuffle[j]}",
    #                         **simDict,
    #                     }
    #                 ),
    #             )

    #     df.to_csv('../outputs/masterOutput/similarities/seedDiff_layer{}.csv'.format(layer))

    # # Compile model similarities
    # modelTypes = ["seedDiff", "itemDiff_max10", "catDiff_max1-10"]

    # # Combine all the csvs for each model type into a single csv
    # for modelType in modelTypes:
    #     # List of csvs to combine
    #     csvs = glob.glob(
    #         f"../outputs/masterOutput/similarities/{modelType}_layer*.csv"
    #     )

    #     # Combine csvs
    #     df = pd.DataFrame()
    #     for csv in csvs:
    #         tmp = pd.read_csv(csv, index_col=0)
    #         # Get layer number from csv name
    #         layer = csv.split("layer")[-1].split(".")[0]

    #         # Add layer to df
    #         tmp["layer"] = layer

    #         # Combine with df
    #         df = pd.concat([df, tmp])

    #     # Save csv
    #     df.to_csv(
    #         f"../outputs/masterOutput/similarities/{modelType}_allLayers.csv",
    #         index=False,
    #     )
