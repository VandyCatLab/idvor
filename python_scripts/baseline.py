import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.datasets import cifar10
import analysis, datasets
import pandas as pd
import os


def yield_transforms(transform, model, layer_idx, dataset):
    """
    Yield transformed representations successfully from the dataset after
    passing through the model to a certain layer_idx. Transforms control what
    is yielded.

    Transform information:
        - 'reflect' yields only one representations
        - 'translate' yields a number a list of four representations equal to
        the smaller dimension, translating the image in all four directions.
        - 'color' yields 51 representations after modifying the color channels
        with multiples of the PCA.
        - 'zoom' yields a number of representations equal to half the smaller
        dimension, zooming into the image.
    """
    # Set model to output reps at selected layer
    inp = model.input
    layer = model.layers[layer_idx]
    out = layer.output
    model = Model(inputs=inp, outputs=out)

    # Get reps for originals
    rep1 = model.predict(dataset, verbose=0, batch_size=224)

    # Reflect and shift don't require remaking the dataset
    if transform == "reflect":
        print(" - Yielding 1 version.", flush=True)
        transDataset = np.flip(dataset, axis=2)
        rep2 = model.predict(transDataset, verbose=0)
        yield 0, rep1, rep2, transDataset

    elif transform == "translate":
        versions = (
            dataset.shape[1]
            if dataset.shape[1] <= dataset.shape[2]
            else dataset.shape[2]
        )

        print(f" - Yielding {versions} versions.")
        for v in tf.range(versions):
            print(f"Translating {v} pixels.", flush=True)
            # Generate transformed imageset
            transImg = tfa.image.translate(dataset, [v, 0])  # Right
            transImg = tf.concat(
                (transImg, tfa.image.translate(dataset, [-v, 0])), axis=0
            )  # Left
            transImg = tf.concat(
                (transImg, tfa.image.translate(dataset, [0, v])), axis=0
            )  # Down
            transImg = tf.concat(
                (transImg, tfa.image.translate(dataset, [0, -v])), axis=0
            )  # Up

            # Get average of all 4 directions
            rep2 = model.predict(transImg, verbose=0, batch_size=224)
            # Split back out
            rep2 = tf.split(rep2, 4)

            yield v, rep1, rep2, transImg

    elif transform == "color":
        versions = 51
        alphas = np.linspace(-1.5, 1.5, versions)

        # print(
        #     "Do PCA on raw training set to get eigenvalues and -vectors",
        #     flush=True,
        # )
        # x_train = datasets.preprocess(datasets.x_trainRaw)
        # x_train = x_train.reshape(-1, 3)
        # cov = np.cov(x_train.T)
        cov = np.cov(dataset.reshape(-1, 3).T)
        values, vectors = np.linalg.eigh(cov)

        print(f" - Yielding {versions} versions.", flush=True)
        for v in range(versions):
            transImg = np.copy(dataset)

            # Add multiple of shift
            alpha = alphas[v]
            print(f"Color shifting alpha: {alpha}.", flush=True)
            change = np.dot(vectors, values * alpha)
            transImg[:, :, :, 0] += change[0]
            transImg[:, :, :, 1] += change[1]
            transImg[:, :, :, 2] += change[2]

            rep2 = model.predict(transImg, verbose=0, batch_size=224)

            yield v, rep1, rep2, transImg

    elif transform == "zoom":
        smallDim = (
            dataset.shape[1]
            if dataset.shape[1] <= dataset.shape[2]
            else dataset.shape[2]
        )
        versions = smallDim // 2

        print(f" - Yielding {versions} versions.", flush=True)
        for v in range(versions):
            print(f"Zooming {v} pixels.", flush=True)
            # Generate transformed imageset
            transformed_dataset = dataset[
                :, v : smallDim - v, v : smallDim - v, :
            ]
            transformed_dataset = tf.image.resize(
                transformed_dataset, (smallDim, smallDim)
            )

            rep2 = model.predict(
                transformed_dataset, verbose=0, batch_size=224
            )

            yield v, rep1, rep2, transformed_dataset

    elif transform == "noise":
        sd = np.std(dataset)
        versions = 20
        alphas = np.linspace(0, 1, versions)

        print(f" - Yielding {versions} versions.", flush=True)
        for a in alphas:
            noise = (
                np.random.normal(loc=0, scale=sd * 3, size=dataset.shape) * a
            )
            transDataset = dataset[:] + noise

            rep2 = model.predict(transDataset, verbose=0, batch_size=224)

            yield a, rep1, rep2, transDataset


def make_dropout_model(model, output_idx, droprate):
    """
    Return a new model with the dropout layers activated during prediction with
    outputs at the list output_idx.
    """
    if output_idx == [-1]:
        output_idx = [len(model.layers) - 2]
    modelInput = model.input

    outputs = []
    inp = model.layers[0].output
    for i, layer in enumerate(model.layers[1 : len(model.layers)]):
        if "dropout" in layer.name:  # Dropout layer
            # Make a new dropout that is used during prediction
            drop = tf.keras.layers.Dropout(droprate)
            inp = drop(inp, training=True)
        else:
            inp = layer(inp)

        if i in output_idx:
            outputs += [inp]

    model = tf.keras.Model(inputs=modelInput, outputs=outputs)
    return model


"""
Sanity check/Visualization functions
"""


def visualize_transform(transform, depth, img_arr):
    if transform == "reflect":
        # Depth doesn't matter
        transformed = np.flip(img_arr, axis=1)
        plt.imshow(transformed)

    elif transform == "color":
        # Depth = alpha value * 1000 (ints -100 : 100)
        alpha = depth / 1000
        img_reshaped = img_arr.reshape(-1, 3)
        cov = np.cov(img_reshaped.T)
        values, vectors = np.linalg.eig(cov)
        change = np.dot(vectors, (values * [alpha, alpha, alpha]).T)
        new_img = np.round(img_arr + change)
        transformed = np.clip(new_img, a_min=0, a_max=255, out=None)
        plt.imshow(transformed)

    elif transform == "zoom":
        dim = img_arr.shape[0]
        v = depth
        img = Image.fromarray(img_arr)
        new_img = img.crop((v, v, dim - v, dim - v))
        new_img = new_img.resize((dim, dim), resample=Image.BICUBIC)
        transformed = img_to_array(new_img)
        plt.imshow(transformed)

    elif transform == "shift":
        dim = img_arr.shape[0]
        v = depth
        empty = np.zeros((dim, dim, 3))
        up_transformed = np.concatenate(
            [img_arr[v:dim, :, :], empty[0:v, :, :]]
        )
        down_transformed = np.concatenate(
            [empty[0:v, :, :], img_arr[0 : dim - v, :, :]]
        )
        left_transformed = np.concatenate(
            [img_arr[:, v:dim, :], empty[:, 0:v, :]], axis=1
        )
        right_transformed = np.concatenate(
            [empty[:, 0:v, :], img_arr[:, 0 : dim - v, :]], axis=1
        )

        _, axarr = plt.subplots(2, 2)
        axarr[0, 0].imshow(up_transformed)
        axarr[0, 1].imshow(down_transformed)
        axarr[1, 0].imshow(left_transformed)
        axarr[1, 1].imshow(right_transformed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Perform baseline analysis, intended to be used in HPC"
    )
    parser.add_argument(
        "--analysis",
        "-a",
        type=str,
        help="type of analysis to run",
        choices=[
            "translate",
            "zoom",
            "reflect",
            "color",
            "dropout",
            "noise",
            "accuracy",
        ],
    )
    parser.add_argument("--model_name", type=str, help="name of model to load")
    parser.add_argument(
        "--model_index",
        "-i",
        type=int,
        help="model index to select weight and shuffle seeds",
    )
    parser.add_argument(
        "--shuffle_seed", type=int, help="shuffle seed of the main model"
    )
    parser.add_argument(
        "--weight_seed", type=int, help="weight seed of the main model"
    )
    parser.add_argument(
        "--model_seeds",
        type=str,
        default="../outputs/masterOutput/modelSeeds.csv",
        help="file location for csv file with model seeds",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="../outputs/masterOutput/dataset.npy",
        help="npy file path for the image dataset to use for analysis",
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        default="../outputs/masterOutput/labels.npy",
        help="npy file path for the labels of the dataset to use for anlaysis.",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="../outputs/masterOutput/models",
        help="directory for all of the models",
    )
    parser.add_argument(
        "--layer_index",
        "-l",
        type=analysis._split_comma_str,
        default="layer indices, split by a comma",
    )
    args = parser.parse_args()

    # Load model
    model, modelName, _ = analysis.get_model_from_args(args)

    if not args.analysis == "accuracy":
        model = analysis.make_allout_model(model)

    # Load dataset
    print("Loading dataset", flush=True)
    dataset = np.load(args.dataset_file)
    print(f"dataset shape: {dataset.shape}", flush=True)

    # Prep analysis functions
    preprocFuns = [
        analysis.preprocess_rsaNumba,
        analysis.preprocess_svcca,
        analysis.preprocess_ckaNumba,
    ]
    simFuns = [
        analysis.do_rsaNumba,
        analysis.do_svcca,
        analysis.do_linearCKANumba,
    ]

    basePath = "../outputs/masterOutput/baseline/"

    if args.analysis in ["translate", "zoom", "reflect", "color", "noise"]:
        simFunNames = [fun.__name__ for fun in simFuns]
        for layer in args.layer_index:
            print(f"Working on layer {layer}.", flush=True)
            # Get transforms generators
            transforms = yield_transforms(
                args.analysis, model, int(layer), dataset
            )

            # Create dataframe
            if args.analysis == "translate":
                directions = (
                    ["right"] * len(simFuns)
                    + ["left"] * len(simFuns)
                    + ["down"] * len(simFuns)
                    + ["up"] * len(simFuns)
                )
                colNames = [
                    f"{fun}-{direct}"
                    for direct, fun in zip(
                        directions, [fun.__name__ for fun in simFuns] * 4
                    )
                ]
                simDf = pd.DataFrame(columns=["version"] + colNames)
            else:
                simDf = pd.DataFrame(columns=["version"] + simFunNames)

            # Get similarity measure per transform
            for v, rep1, rep2, _ in transforms:
                if args.analysis == "translate":
                    # Calculate similarity for each direction
                    simDirs = []
                    for rep in rep2:
                        rep = np.array(rep)
                        simDirs += [
                            analysis.multi_analysis(
                                rep1, rep, preprocFuns, simFuns
                            )
                        ]

                    # Save all directions
                    tmp = [v.numpy()]
                    for dic in simDirs:
                        tmp += [dic[key] for key in dic.keys()]

                    simDf.loc[len(simDf.index)] = tmp
                else:
                    sims = analysis.multi_analysis(
                        rep1, rep2, preprocFuns, simFuns
                    )
                    simDf.loc[len(simDf.index)] = [v] + [
                        sims[fun] for fun in sims.keys()
                    ]

            # Save
            outPath = os.path.join(
                basePath, f"{modelName[0:-3]}l{layer}-{args.analysis}.csv"
            )
            simDf.to_csv(outPath, index=False)

    elif args.analysis == "accuracy":
        augList = ["translate", "zoom", "reflect", "color", "noise"]
        dataLabels = np.load(args.labels_file)

        # First handle augment tests first
        for aug in augList:
            # Make transforms, note selecting first layer for efficiency sake
            transforms = yield_transforms(aug, model, 1, dataset)

            # Create dataframe
            colNames = (
                ["version", "rightAcc", "leftAcc", "downAcc", "upAcc"]
                if aug == "translate"
                else ["version", "acc"]
            )
            accDF = pd.DataFrame(columns=colNames)

            # Get similarity measure per transform
            for v, _, _, transImg in transforms:
                if aug == "translate":
                    # Split image to the directions
                    transDir = tf.split(transImg, 4)
                    accs = [0.0] * 4
                    for i, direct in enumerate(transDir):
                        _, accs[i] = model.evaluate(
                            direct, dataLabels, batch_size=224
                        )

                    accDF.loc[len(accDF.index)] = [float(v.numpy())] + accs
                else:
                    _, acc = model.evaluate(
                        transImg, dataLabels, batch_size=224
                    )
                    accDF.loc[len(accDF.index)] = [v, acc]

            # Save
            outPath = os.path.join(
                basePath, f"{modelName[0:-3]}-acc-{aug}.csv"
            )
            accDF.to_csv(outPath, index=False)

        # Now do dropout
        dropRates = np.arange(0, 1, 0.05)
        accDF = pd.DataFrame(columns=["version", "acc"])
        for drop in dropRates:
            dropModel = make_dropout_model(model, [-1], drop)
            dropModel.compile(
                optimizer="SGD",
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            _, acc = dropModel.evaluate(dataset, dataLabels, batch_size=224)
            accDF.loc[len(accDF.index)] = [drop, acc]

        outPath = os.path.join(basePath, f"{modelName[0:-3]}-acc-drop.csv")
        accDF.to_csv(outPath, index=False)
    elif args.analysis == "dropout":
        layerIdx = [int(idx) for idx in args.layer_index]
        layerIdx.sort()
        dropRates = np.arange(0, 1, 0.05)

        data = pd.DataFrame(
            columns=[fun.__name__ for fun in simFuns] + ["layer", "dropRate"]
        )
        for drop in dropRates:
            dropModel = make_dropout_model(model, layerIdx, drop)
            rep1 = dropModel.predict(dataset)
            rep2 = dropModel.predict(dataset)

            # Make sure it's a list
            rep1 = rep1 if isinstance(rep1, list) else [rep1]
            rep2 = rep2 if isinstance(rep1, list) else [rep2]

            for i in range(len(rep1)):
                sims = analysis.multi_analysis(
                    rep1[i], rep2[i], preprocFuns, simFuns
                )
                sims["layer"] = layerIdx[i]
                sims["dropRate"] = drop
                data.loc[len(data.index)] = [sims[fun] for fun in sims.keys()]

        outPath = os.path.join(
            basePath,
            f"{modelName[0:-3]}-dropout-{','.join([str(idx) for idx in layerIdx])}.csv",
        )
        data.to_csv(outPath, index=False)
