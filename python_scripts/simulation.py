import analysis
import tensorflow as tf
import numpy as np
import pandas as pd
import os


def permuteTest(
    model,
    dataset,
    nImgs=None,
    outputPath=None,
    preprocFuns=None,
    simFuns=None,
    analysisNames=None,
    outputIdx=-2,
    nPermutes=1000,
):

    # Subset dataset
    if nImgs is not None:
        dataset = dataset[:nImgs]

    # Set model to output reps at layer
    inp = model.input
    layer = model.layers[outputIdx]
    out = layer.output

    tmpModel = tf.keras.Model(inputs=inp, outputs=out)

    # Get reps for originals and flatten
    rep_orig = tmpModel.predict(dataset)
    repShape = rep_orig.shape
    rep_flat = rep_orig.flatten()
    repSD = np.std(rep_flat)

    if outputPath is None:
        permutePath = "../outputs/masterOutput/permuteSims.csv"
    else:
        permutePath = outputPath

    if os.path.exists(permutePath):
        print("Using existing permutation test results")
        permuteData = pd.read_csv(permutePath)
    else:
        colNames = analysisNames + ["analysis"]

        permuteData = pd.DataFrame(columns=colNames)

        # Random permutation
        print("Doing random simulation")
        df = pd.DataFrame(columns=colNames, index=range(nPermutes))
        for permute in range(nPermutes):
            if permute % 100 == 0:
                print(f"-- Permutation at {permute}")

            rep1 = np.random.choice(rep_flat, size=repShape, replace=False)
            rep2 = np.random.choice(rep_flat, size=repShape, replace=False)

            sims = analysis.multi_analysis(
                rep1, rep2, preprocFuns, simFuns, names=analysisNames
            )
            df.loc[permute] = list(sims.values()) + ["random"]
        permuteData = permuteData.append(df)

        # Noised permutation
        print("Doing noise simulation")
        df = pd.DataFrame(columns=colNames, index=range(nPermutes))
        for permute in range(nPermutes):
            if permute % 100 == 0:
                print(f"-- Permutation at {permute}")

            rep = np.random.choice(rep_flat, size=repShape, replace=False)
            repNoise = rep + np.random.normal(scale=repSD, size=repShape)
            repNoise = repNoise.astype(np.float32)

            sims = analysis.multi_analysis(
                rep, repNoise, preprocFuns, simFuns, names=analysisNames
            )
            df.loc[permute] = list(sims.values()) + ["noise"]
        permuteData = permuteData.append(df)

        # Cut a neuron
        print("Doing ablation simulation")
        df = pd.DataFrame(columns=colNames, index=range(nPermutes))
        for permute in range(nPermutes):
            if permute % 100 == 0:
                print(f"-- Permutation at {permute}")

            rep = np.random.choice(rep_flat, size=repShape, replace=False)
            repCut = np.copy(rep)
            nFeatures = rep.shape[1]
            toCut = int(np.round(nFeatures * 0.1))
            repCut[
                :, np.random.choice(nFeatures, size=toCut, replace=False)
            ] = 0

            sims = analysis.multi_analysis(
                rep, repCut, preprocFuns, simFuns, names=analysisNames
            )
            df.loc[permute] = list(sims.values()) + ["ablate"]
        permuteData = permuteData.append(df)
        permuteData.to_csv(permutePath)


def sizeRatioTest(
    nImgs=None,
    outputPath=None,
    model=None,
    imgset=None,
    preprocFuns=None,
    simFuns=None,
    analysisNames=None,
    outputIdx=-2,
    nPermutes=1000,
):
    # load dataset
    if imgset is None:
        imgset = np.load("../outputs/masterOutput/dataset.npy")
    else:
        imgset = np.load(imgset)

    # Subset dataset
    if nImgs is not None:
        imgset = imgset[:nImgs]

    if model is None:
        modelPath = "../outputs/masterOutput/models/w0s0.pb"
        model = tf.keras.models.load_model(modelPath)
        model = tf.keras.Model(
            inputs=model.input, outputs=model.layers[outputIdx].output
        )
    rep = model.predict(imgset).flatten()

    colNames = analysisNames + ["sample", "features"]

    permuteSims = pd.DataFrame(columns=colNames)

    nMax = imgset.shape[0]
    ratiosRange = np.arange(0.05, 2, 0.05)

    for ratio in ratiosRange:
        print(f"Analyzing img:samples ratio: {ratio}")
        repShape = (nMax, int(nMax * ratio))
        for permute in range(nPermutes):
            if permute % 100 == 0:
                print(f"Permutation at {permute}")

            # Sample reps
            rep1 = np.random.choice(rep, size=repShape)
            rep2 = np.random.choice(rep, size=repShape)

            sims = analysis.multi_analysis(
                rep1, rep2, preprocFuns, simFuns, names=analysisNames
            )
            sims["sample"] = repShape[0]
            sims["features"] = repShape[1]
            permuteSims = pd.concat(
                (
                    permuteSims,
                    pd.DataFrame.from_dict(sims, orient="index").transpose(),
                )
            )

    if outputPath is None:
        outPath = "../outputs/masterOutput/ratioSims.csv"
    else:
        outPath = outputPath
    permuteSims.to_csv(outPath, index=False)


def bigSizeRatioTest():
    # load dataset
    imgset = np.load("../outputs/masterOutput/bigDataset.npy")
    model = tf.keras.applications.MobileNetV3Small(input_shape=(224, 224, 3))
    model.compile(metrics=["top_k_categorical_accuracy"])
    print(f"Model loaded: MobileNetV3Small", flush=True)
    rep = model.predict(imgset).flatten()

    preprocFuns = [
        analysis.preprocess_peaRsaNumba,
        analysis.preprocess_eucRsaNumba,
        analysis.preprocess_speRsaNumba,
        analysis.preprocess_svcca,
        analysis.preprocess_ckaNumba,
    ]
    simFuns = [
        analysis.do_rsaNumba,
        analysis.do_rsaNumba,
        analysis.do_rsaNumba,
        analysis.do_svcca,
        analysis.do_linearCKANumba,
    ]
    analysisNames = ["peaRsa", "eucRsa", "speRsa", "svcca", "cka"]
    colNames = analysisNames + ["sample", "features"]

    permuteSims = pd.DataFrame(columns=colNames)

    nMax = imgset.shape[0]
    ratiosRange = np.arange(0.05, 1, 0.05)
    nPermute = 100

    for ratio in ratiosRange:
        print(f"Analyzing img:samples ratio: {ratio}")
        repShape = (nMax, int(nMax * ratio))
        for permute in range(nPermute):
            if permute % 10 == 0:
                print(f"Permutation at {permute}")

            # Sample reps
            rep1 = np.random.choice(rep, size=repShape)
            rep2 = np.random.choice(rep, size=repShape)

            sims = analysis.multi_analysis(
                rep1, rep2, preprocFuns, simFuns, names=analysisNames
            )
            sims["sample"] = repShape[0]
            sims["features"] = repShape[1]
            permuteSims = pd.concat(
                (
                    permuteSims,
                    pd.DataFrame.from_dict(sims, orient="index").transpose(),
                )
            )
    permuteSims.to_csv("../outputs/masterOutput/bigRatioSims.csv", index=False)


def parametricAblation(
    model,
    imgset,
    nImgs=None,
    outputPath=None,
    preprocFuns=None,
    simFuns=None,
    analysisNames=None,
    outputIdx=-2,
    nPermutes=1000,
):

    # Subset dataset
    if nImgs is not None:
        imgset = imgset[:nImgs]

    # Set model to output reps at layer
    inp = model.input
    layer = model.layers[outputIdx]
    out = layer.output

    tmpModel = tf.keras.Model(inputs=inp, outputs=out)

    # Get reps for originals and flatten
    rep_orig = tmpModel.predict(imgset)
    repShape = rep_orig.shape
    rep_flat = rep_orig.flatten()

    # Calculate # of neurons to keep
    neuronsKept = np.int32(np.arange(0.3, 1.1, 0.1) * repShape[1])

    colNames = analysisNames + ["Neurons"]

    permuteData = pd.DataFrame(columns=colNames)
    if outputPath is None:
        outPath = "../outputs/masterOutput/ablateSims.csv"
    else:
        outPath = outputPath
    print("Doing ablation simulation")
    for permute in range(nPermutes):
        if permute % 100 == 0:
            print(f"-- Permutation at {permute}")

        df = pd.DataFrame(columns=colNames, index=neuronsKept)
        for nNeurons in neuronsKept:
            rep = np.random.choice(rep_flat, size=repShape, replace=False)
            repCut = np.copy(rep)
            repCut = repCut[
                :, np.random.choice(rep.shape[1], size=nNeurons, replace=False)
            ]

            sims = analysis.multi_analysis(
                rep, repCut, preprocFuns, simFuns, names=analysisNames
            )
            df.loc[nNeurons] = list(sims.values()) + [nNeurons]
        permuteData = permuteData.append(df)
    permuteData.to_csv(outPath)


def parametricNoise(
    minNoise=0.0,
    maxNoise=1,
    step=0.1,
    permutations=10000,
    seed=None,
    nImgs=None,
    outputPath=None,
    preprocFuns=None,
    simFuns=None,
    analysisNames=None,
    outputIdx=-2,
):
    modelPath = "../outputs/masterOutput/models/w0s0.pb"
    print("Loading model")
    model = tf.keras.models.load_model(modelPath)

    # load dataset
    imgset = np.load("../outputs/masterOutput/dataset.npy")

    # Only use a subset of the images
    if nImgs is not None:
        imgset = imgset[:nImgs]

    # Dataset information
    num_imgs = imgset.shape[0]
    dim = imgset.shape[1]
    correlations = []

    # Set model to output reps at layer
    inp = model.input
    layer = model.layers[outputIdx]
    out = layer.output

    tmpModel = tf.keras.Model(inputs=inp, outputs=out)

    # Get reps for originals and flatten
    rep_orig = tmpModel.predict(imgset)
    repShape = rep_orig.shape
    rep_flat = rep_orig.flatten()
    repMean = np.mean(rep_flat)
    repSD = np.std(rep_flat)

    colNames = analysisNames + ["Noise"]

    permuteData = pd.DataFrame(columns=colNames)
    noiseRange = np.arange(minNoise, maxNoise + 0.1, step)

    # Set seed if available
    if seed is not None:
        print(f"Setting seed {seed}", flush=True)
        np.random.seed(seed)
        outPath = f"../outputs/masterOutput/noiseSims{seed}.csv"
    else:
        if outputPath is None:
            outPath = "../outputs/masterOutput/noiseSims.csv"
        else:
            outPath = outputPath
    print("Doing parametric noise simulation")
    for permute in range(permutations):
        if permute % 100 == 0:
            print(f"-- Permutation at {permute}", flush=True)

        rep = np.random.choice(rep_flat, size=repShape, replace=False)

        df = pd.DataFrame(columns=colNames, index=noiseRange)
        for noise in noiseRange:
            repNoise = rep + np.random.normal(
                scale=repSD * noise, size=repShape
            )
            repNoise = repNoise.astype(np.float32)

            sims = analysis.multi_analysis(
                rep,
                repNoise,
                preprocFuns,
                simFuns,
                verbose=False,
                names=analysisNames,
            )
            df.loc[noise] = list(sims.values()) + [noise]
        permuteData = permuteData.append(df)

    permuteData.to_csv(outPath)


def sanity_check(
    model,
    dataset,
    nImgs=None,
    outputPath=None,
    preprocFuns=None,
    simFuns=None,
    analysisNames=None,
    outputIdx=-2,
    nPermutes=1000,
):
    # Subset dataset
    if nImgs is not None:
        dataset = dataset[:nImgs]

    # Set model to output reps at layer
    inp = model.input
    layer = model.layers[outputIdx]
    out = layer.output

    tmpModel = tf.keras.Model(inputs=inp, outputs=out)

    # Get reps for originals and flatten
    rep_orig = tmpModel.predict(dataset)
    repShape = rep_orig.shape
    rep_flat = rep_orig.flatten()

    colNames = analysisNames

    permuteData = pd.DataFrame(columns=colNames, index=range(nPermutes))

    if outputPath is None:
        outPath = "../outputs/masterOutput/sanityCheckSims.csv"
    else:
        outPath = outputPath
    print("Performing sanity check, permuting features.")
    for permute in range(nPermutes):
        if permute % 100 == 0:
            print(f"-- Permutation at {permute}")

        rep = np.random.choice(rep_flat, size=repShape, replace=False)
        repPermute = np.copy(rep)
        repPermute = repPermute[
            :, np.random.choice(rep.shape[1], size=repShape[1], replace=False)
        ]

        sims = analysis.multi_analysis(
            rep, repPermute, preprocFuns, simFuns, names=analysisNames
        )
        permuteData.loc[permute] = list(sims.values())

        permuteData.to_csv(outPath)


if __name__ == "__main__":
    # Setup argparse
    import argparse

    parser = argparse.ArgumentParser(
        description="Do some simulation anlaysis, inteded to be used on HPC"
    )
    parser.add_argument(
        "--analysis",
        "-a",
        type=str,
        required=True,
        help="analysis to perform",
        choices=["noise", "simulations", "sanity", "ablate", "sizeRatio"],
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        help="seed for random number generator",
        default=None,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="model to use for representations",
        choices=["allcnnc", "vgg", "resnet"],
        default="allcnnc",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="dataset to use",
        choices=["cifar10", "imagenet"],
        default="cifar10",
    )
    parser.add_argument(
        "--nImgs",
        type=int,
        default=None,
        help="number images from the dataset to use",
    )
    parser.add_argument(
        "--outputPath",
        type=str,
        default=None,
        help="path to output file, if not specified will use default",
    )
    parser.add_argument(
        "--simSet",
        type=str,
        default="all",
        help="which set of similarity functions to use",
    )
    parser.add_argument(
        "--outputIdx",
        type=int,
        default=-2,
        help="index of layer to output from",
    )
    parser.add_argument(
        "--permutes",
        type=int,
        default=1000,
        help="number of permutations to do",
    )
    args = parser.parse_args()

    preprocFuns, simFuns, analysisNames = analysis.get_funcs(args.simSet)

    if args.analysis == "noise":
        parametricNoise(
            maxNoise=4.0,
            step=0.01,
            permutations=args.permutes,
            seed=args.seed,
            nImgs=args.nImgs,
            outputPath=args.outputPath,
            preprocFuns=preprocFuns,
            simFuns=simFuns,
            analysisNames=analysisNames,
            outputIdx=args.outputIdx,
        )
    elif args.analysis == "simulations":
        if args.dataset == "cifar10":
            dataset = np.load("../outputs/masterOutput/dataset.npy")
        elif args.dataset == "imagenet":
            dataset = np.load("../outputs/masterOutput/bigDataset.npy")

        if args.model == "allcnnc":
            modelPath = "../outputs/masterOutput/models/w0s0.pb"
            model = tf.keras.models.load_model(modelPath)
        elif args.model == "vgg":
            model = tf.keras.applications.vgg16.VGG16(
                input_shape=(224, 224, 3)
            )
            model.compile(metrics=["top_k_categorical_accuracy"])
        elif args.model == "resnet":
            model = tf.keras.applications.resnet50.ResNet50(
                input_shape=(224, 224, 3)
            )
            model.compile(metrics=["top_k_categorical_accuracy"])

        permuteTest(
            dataset=dataset,
            model=model,
            nImgs=args.nImgs,
            outputPath=args.outputPath,
            preprocFuns=preprocFuns,
            simFuns=simFuns,
            analysisNames=analysisNames,
            outputIdx=args.outputIdx,
            nPermutes=args.permutes,
        )
    elif args.analysis == "sanity":
        if args.dataset == "cifar10":
            dataset = np.load("../outputs/masterOutput/dataset.npy")
        elif args.dataset == "imagenet":
            dataset = np.load("../outputs/masterOutput/bigDataset.npy")

        if args.model == "allcnnc":
            modelPath = "../outputs/masterOutput/models/w0s0.pb"
            model = tf.keras.models.load_model(modelPath)
        elif args.model == "vgg":
            model = tf.keras.applications.vgg16.VGG16(
                input_shape=(224, 224, 3)
            )
            model.compile(metrics=["top_k_categorical_accuracy"])
        elif args.model == "resnet":
            model = tf.keras.applications.resnet50.ResNet50(
                input_shape=(224, 224, 3)
            )
            model.compile(metrics=["top_k_categorical_accuracy"])

        sanity_check(
            model=model,
            dataset=dataset,
            nImgs=args.nImgs,
            outputPath=args.outputPath,
            preprocFuns=preprocFuns,
            simFuns=simFuns,
            analysisNames=analysisNames,
            outputIdx=args.outputIdx,
            nPermutes=args.permutes,
        )
    elif args.analysis == "ablate":
        if args.dataset == "cifar10":
            dataset = "../outputs/masterOutput/dataset.npy"
        elif args.dataset == "imagenet":
            dataset = "../outputs/masterOutput/bigDataset.npy"

        if args.model == "allcnnc":
            modelPath = "../outputs/masterOutput/models/w0s0.pb"
            model = tf.keras.models.load_model(modelPath)
        elif args.model == "vgg":
            model = tf.keras.applications.vgg16.VGG16(
                input_shape=(224, 224, 3)
            )
            model.compile(metrics=["top_k_categorical_accuracy"])
        elif args.model == "resnet":
            model = tf.keras.applications.resnet50.ResNet50(
                input_shape=(224, 224, 3)
            )
            model.compile(metrics=["top_k_categorical_accuracy"])

        dataset = np.load(dataset)

        parametricAblation(
            model=model,
            imgset=dataset,
            nImgs=args.nImgs,
            outputPath=args.outputPath,
            preprocFuns=preprocFuns,
            simFuns=simFuns,
            analysisNames=analysisNames,
            outputIdx=args.outputIdx,
            nPermutes=args.permutes,
        )
    elif args.analysis == "sizeRatio":
        if args.dataset == "cifar10":
            dataset = "../outputs/masterOutput/dataset.npy"
        elif args.dataset == "imagenet":
            dataset = "../outputs/masterOutput/bigDataset.npy"

        if args.model == "allcnnc":
            modelPath = "../outputs/masterOutput/models/w0s0.pb"
            model = tf.keras.models.load_model(modelPath)
        elif args.model == "vgg":
            model = tf.keras.applications.vgg16.VGG16(
                input_shape=(224, 224, 3)
            )
            model.compile(metrics=["top_k_categorical_accuracy"])
        elif args.model == "resnet":
            model = tf.keras.applications.resnet50.ResNet50(
                input_shape=(224, 224, 3)
            )
            model.compile(metrics=["top_k_categorical_accuracy"])

        sizeRatioTest(
            model=model,
            nImgs=args.nImgs,
            outputPath=args.outputPath,
            imgset=dataset,
            preprocFuns=preprocFuns,
            simFuns=simFuns,
            analysisNames=analysisNames,
            outputIdx=args.outputIdx,
            nPermutes=args.permutes,
        )
    else:
        raise ValueError(f"Unknown analysis: {args.analysis}")
