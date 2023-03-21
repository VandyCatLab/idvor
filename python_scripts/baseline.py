import re
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch
import tensorflow_addons as tfa
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.datasets import cifar10
import analysis, datasets
import pandas as pd
import os
from scipy.stats import norm


def yield_transforms(
    transform,
    model,
    layer_idx,
    dataset,
    return_aug=True,
    versions=None,
    slice=None,
):
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
        - 'magAux' yields a number a list of four representations equal to
        the smaller dimension, translating the image in all four directions
        after reflecting.
    """

    def batched_call(model, input, batch_size):
        # Get counts
        nImages = input.shape[0]
        completeBatches = (
            nImages // batch_size if batch_size < input.shape[0] else input.shape[0]
        )
        finalBatchSize = nImages % batch_size

        # Loop through batches
        outs = []
        for idx in range(completeBatches):
            # Create a batch
            with tf.device("/cpu:0"):
                batch = input[(idx * batch_size) : ((idx + 1) * batch_size), :, :, :]
            outs += [model.call(batch, training=False)]

        # Final batch
        with tf.device("/cpu:0"):
            batch = input[
                ((idx + 1) * batch_size) : (
                    ((idx + 1) * batch_size) + finalBatchSize + 1
                ),
                :,
                :,
                :,
            ]
        outs += [model.call(batch, training=False)]

        return np.concatenate(outs)

    # Set model to output reps at selected layer
    inp = model.input
    layer = model.layers[layer_idx]
    out = layer.output
    model = Model(inputs=inp, outputs=out)

    # Turn dataset into tensor on cpu to avoid memory problems
    with tf.device("/cpu:0"):
        dataset = tf.convert_to_tensor(dataset)

    # Get reps for originals
    rep1 = model.predict(dataset, verbose=0, batch_size=128)

    # Reflect and shift don't require remaking the dataset
    if transform == "reflect":
        print(" - Yielding 1 version.", flush=True)
        with tf.device("/cpu:0"):
            transDataset = tf.image.flip_left_right(dataset)
        rep2 = model.predict(transDataset, verbose=0, batch_size=512)
        if return_aug:
            yield 0, rep1, rep2, transDataset
        else:
            yield 0, rep1, rep2

    elif transform == "translate":
        if versions is None:
            versions = (
                dataset.shape[1]
                if dataset.shape[1] <= dataset.shape[2]
                else dataset.shape[2]
            )

        print(f" - Yielding {versions} versions.")
        for v in tf.range(versions):
            if slice is not None and not slice == v.numpy():
                continue
            print(f"Translating {v} pixels.", flush=True)

            # Generate transformed imageset
            with tf.device("/cpu:0"):
                transImg = tfa.image.translate(dataset, [v, 0])  # Right
            rep2 = [batched_call(model, transImg, 512)]

            with tf.device("/cpu:0"):
                transImg = tfa.image.translate(dataset, [-v, 0])  # Left
            rep2 += [batched_call(model, transImg, 512)]

            with tf.device("/cpu:0"):
                transImg = tfa.image.translate(dataset, [0, v])  # Down
            rep2 += [batched_call(model, transImg, 512)]

            with tf.device("/cpu:0"):
                transImg = tfa.image.translate(dataset, [0, -v])  # Up
            rep2 += [batched_call(model, transImg, 512)]

            if return_aug:
                # Regenerate translated dataset and concatenate all four directions
                with tf.device("/cpu:0"):
                    transImg = tf.concat(
                        [
                            tfa.image.translate(dataset, [v, 0]),
                            tfa.image.translate(dataset, [-v, 0]),
                            tfa.image.translate(dataset, [0, v]),
                            tfa.image.translate(dataset, [0, -v]),
                        ],
                        axis=0,
                    )
                yield v, rep1, rep2, transImg
            else:
                yield v, rep1, rep2

    elif transform == "color":
        if versions is None:
            versions = 51
        alphas = np.linspace(-1.5, 1.5, versions)

        # print(
        #     "Do PCA on raw training set to get eigenvalues and -vectors",
        #     flush=True,
        # )
        # x_train = datasets.preprocess(datasets.x_trainRaw)
        # x_train = x_train.reshape(-1, 3)
        # cov = np.cov(x_train.T)
        with tf.device("/cpu:0"):
            cov = np.cov(tf.transpose(tf.reshape(dataset, (-1, 3))))
        values, vectors = np.linalg.eigh(cov)

        print(f" - Yielding {versions} versions.", flush=True)
        for v in range(versions):
            if slice is not None and not slice == v:
                continue
            # Add multiple of shift
            alpha = alphas[v]
            print(f"Color shifting alpha: {alpha}.", flush=True)
            change = np.dot(vectors, values * alpha)
            with tf.device("/cpu:0"):
                changes = tf.zeros(1)
                changes = tf.stack(
                    [
                        tf.zeros(dataset.shape[0:3]) + change[0],
                        tf.zeros(dataset.shape[0:3]) + change[1],
                        tf.zeros(dataset.shape[0:3]) + change[2],
                    ],
                    axis=-1,
                )
                changes = tf.cast(changes, dataset.dtype)
                transImg = dataset + changes

            rep2 = batched_call(model, transImg, 512)

            if return_aug:
                yield v, rep1, rep2, transImg
            else:
                yield v, rep1, rep2

    elif transform == "zoom":
        smallDim = (
            dataset.shape[1]
            if dataset.shape[1] <= dataset.shape[2]
            else dataset.shape[2]
        )

        if versions is None:
            versions = smallDim // 2

        print(f" - Yielding {versions} versions.", flush=True)
        for v in range(versions):
            if slice is not None and not slice == v:
                continue
            print(f"Zooming {v} pixels.", flush=True)
            # Generate transformed imageset
            with tf.device("/cpu:0"):
                transformed_dataset = tf.zeros(1)
                transformed_dataset = dataset[:, v : smallDim - v, v : smallDim - v, :]
                transformed_dataset = tf.image.resize(
                    transformed_dataset, (smallDim, smallDim)
                )

            rep2 = batched_call(model, transformed_dataset, 512)

            if return_aug:
                yield v, rep1, rep2, transformed_dataset
            else:
                yield v, rep1, rep2

    elif transform == "noise":
        sd = np.std(dataset)
        if versions is None:
            versions = 20
        alphas = np.linspace(0, 1, versions)

        print(f" - Yielding {versions} versions.", flush=True)
        for a in alphas:
            if slice is not None and not slice == a:
                continue
            with tf.device("/cpu:0"):
                noise = tf.zeros(1)
                noise = tf.random.normal(
                    shape=dataset.shape, stddev=sd * a, dtype=dataset.dtype
                )
                transDataset = dataset + noise

            rep2 = batched_call(model, transDataset, 512)

            if return_aug:
                yield a, rep1, rep2, transDataset
            else:
                yield a, rep1, rep2

    elif transform == "maxAug":
        print(f" - Yielding 1 version.")
        v = 5 if versions is None else versions
        print(
            f"Translating {v} pixels in both directions after horizontal flip.",
            flush=True,
        )
        with tf.device("/cpu:0"):
            dataset = tf.image.flip_left_right(dataset)

        # Generate transformed imageset
        with tf.device("/cpu:0"):
            transImg = tfa.image.translate(dataset, [v, v])
        rep2 = [batched_call(model, transImg, 512)]

        with tf.device("/cpu:0"):
            transImg = tfa.image.translate(dataset, [-v, -v])
        rep2 += [batched_call(model, transImg, 512)]

        with tf.device("/cpu:0"):
            transImg = tfa.image.translate(dataset, [v, -v])
        rep2 += [batched_call(model, transImg, 512)]

        with tf.device("/cpu:0"):
            transImg = tfa.image.translate(dataset, [-v, v])
        rep2 += [batched_call(model, transImg, 512)]

        if return_aug:
            # Regenerate translated dataset and concatenate all four directions
            with tf.device("/cpu:0"):
                transImg = tf.concat(
                    [
                        tfa.image.translate(dataset, [v, v]),
                        tfa.image.translate(dataset, [-v, -v]),
                        tfa.image.translate(dataset, [v, -v]),
                        tfa.image.translate(dataset, [-v, v]),
                    ],
                    axis=0,
                )
            yield v, rep1, rep2, transImg
        else:
            yield v, rep1, rep2

    elif transform == "random":
        print(f" - Yielding {versions} repeats.")
        for v in range(versions):
            with tf.device("/cpu:0"):
                transImg = tf.identity(dataset)

                # Flip half of the images
                # flipUniform = tf.random.uniform(
                #     shape=[dataset.shape[0], 1, 1, 1],
                # )
                # transImg = tf.where(
                #     tf.less(flipUniform, 0.5),
                #     tf.image.flip_left_right(transImg),
                #     transImg,
                # )

                transImg = tf.cond(
                    tf.less(tf.random.uniform(shape=(), minval=0, maxval=1), 0.5),
                    lambda: tf.image.flip_left_right(transImg),
                    lambda: transImg,
                )

                # Randomly translate images
                # transUniform = tf.random.uniform(
                #     shape=(dataset.shape[0], 2), minval=-0, maxval=0
                # )
                transUniform = tf.random.uniform(shape=(1, 2), minval=-5, maxval=5)
                transImg = tfa.image.translate(transImg, transUniform)

                rep2 = batched_call(model, transImg, 512)

                if return_aug:
                    yield v, rep1, rep2, transImg
                else:
                    yield v, rep1, rep2


def yield_big_transforms(
    transform,
    model,
    preproc_fun,
    layer_idx,
    dataset,
    versions=100,
):
    """Unlike previous function, this one acts on preprocessed images array."""
    origShape = dataset.shape
    with tf.device("/cpu:0"):
        ds = tf.keras.layers.CenterCrop(224, 224)(dataset)
        df = preproc_fun(ds)

    # Set model to output reps at selected layer
    inp = model.input
    layer = model.layers[layer_idx]
    out = layer.output
    model = Model(inputs=inp, outputs=out)

    # Get reps for originals
    rep1 = model.predict(ds, verbose=0)

    eigVals = np.array([115.25870013, 35.37227674, 17.20782363])
    eigVecs = np.array(
        [
            [-0.58215351, 0.69303716, -0.42520205],
            [-0.58321022, 0.00846138, 0.81227719],
            [-0.56653607, -0.72085221, -0.39926053],
        ]
    )

    if transform == "maxAug":
        # Calculate extreme color shift values
        colorMax = np.matmul(eigVecs, norm.ppf(0.975, loc=0, scale=0.1) * eigVals)
        colorMax = np.concatenate(
            [
                np.tile(colorMax[0], (224, 224, 1)),
                np.tile(colorMax[1], (224, 224, 1)),
                np.tile(colorMax[2], (224, 224, 1)),
            ],
            axis=2,
        )

        def _load_images_maxAug(path):
            img = tf.keras.preprocessing.image.smart_resize(img, (224, 224))
            img = preproc_fun(img)

            # Flip image
            img = tf.image.flip_left_right(img)

        raise NotImplementedError

    elif transform == "random":
        print(f" - Yielding {versions} repeats.")
        transDs = tf.data.Dataset.list_files(dataset + "/*.png", shuffle=False)

        def _load_images_aug(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=3)

            # Maybe flip image
            img = tf.image.random_flip_left_right(img)

            # Random resize and crop
            randSize = tf.random.uniform(
                shape=[], minval=256, maxval=512, dtype=tf.int32
            )

            img = tf.cond(
                tf.greater(tf.shape(img)[0], tf.shape(img)[1]),
                lambda: tf.image.resize(
                    img, (10000, randSize), preserve_aspect_ratio=True
                ),
                lambda: tf.image.resize(
                    img, (randSize, 10000), preserve_aspect_ratio=True
                ),
            )

            img = tf.image.random_crop(img, (224, 224, 3))

            # Preprocess
            img = preproc_fun(img)

            # Add batch dimension
            img = tf.expand_dims(img, axis=0)

            return img

        transDs = transDs.map(_load_images_aug)
        transDs = transDs.batch(32)

        for v in range(versions):
            # Get reps for transformed images
            rep2 = model.predict(transDs, verbose=0)

            yield v, rep1, rep2


def make_dropout_model(model, output_idx, droprate):
    """
    Return a new model with the dropout layers activated during prediction with
    outputs at the list output_idx.
    """
    # Fix negative indices
    newOutIdx = []
    for idx in output_idx:
        newOutIdx += [idx % len(model.layers)]
    output_idx = newOutIdx
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
        up_transformed = np.concatenate([img_arr[v:dim, :, :], empty[0:v, :, :]])
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
            "maxAug",
            "random",
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
    parser.add_argument("--weight_seed", type=int, help="weight seed of the main model")
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
        "--model_dir",
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
    parser.add_argument(
        "--group", "-g", type=str, help="analysis group to store together"
    )
    parser.add_argument(
        "--versions",
        "-v",
        type=int,
        help="number of augmentation versions",
        default=None,
    )
    parser.add_argument(
        "--version_slice",
        type=float,
        help="If set, instead of sweep across versions, only return a single version slice",
        default=None,
    )
    parser.add_argument(
        "--sim_funs",
        type=str,
        default="all",
        help="which similarity functions to use",
    )
    args = parser.parse_args()

    if args.analysis is not None:
        # Load model
        model, modelName, _ = analysis.get_model_from_args(args)

        if not args.analysis == "accuracy":
            model = analysis.make_allout_model(model)

        # Load dataset
        print("Loading dataset", flush=True)
        dataset = np.load(args.dataset_file)
        print(f"dataset shape: {dataset.shape}", flush=True)

        # Prep analysis functions
        preprocFuns, simFuns, analysisNames = analysis.get_funcs(args.sim_funs)

        basePath = "../outputs/masterOutput/baseline/"

        # Add analysis group
        if args.group is not None:
            basePath += args.group + "/"
            if not os.path.exists(basePath):
                os.mkdir(basePath)

        if args.analysis in ["translate", "zoom", "reflect", "color", "noise"]:
            for layer in args.layer_index:
                print(f"Working on layer {layer}.", flush=True)
                # Get transforms generators
                transforms = yield_transforms(
                    args.analysis,
                    model,
                    int(layer),
                    dataset,
                    return_aug=False,
                    versions=args.versions,
                    slice=args.version_slice,
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
                        for direct, fun in zip(directions, analysisNames * 4)
                    ]
                    simDf = pd.DataFrame(columns=["version"] + colNames)
                else:
                    simDf = pd.DataFrame(columns=["version"] + analysisNames)

                outPath = os.path.join(
                    basePath,
                    f"{modelName.split('.')[0]}l{layer}-{args.analysis}{'-v'+str(args.version_slice) if args.version_slice is not None else ''}.csv",
                )

                if not os.path.exists(outPath):
                    # Get similarity measure per transform
                    for v, rep1, rep2 in transforms:
                        if args.analysis == "translate":
                            # Calculate similarity for each direction
                            simDirs = []
                            for rep in rep2:
                                rep = np.array(rep)
                                simDirs += [
                                    analysis.multi_analysis(
                                        rep1,
                                        rep,
                                        preprocFuns,
                                        simFuns,
                                        analysisNames,
                                    )
                                ]

                            # Save all directions
                            tmp = [v.numpy()]
                            for dic in simDirs:
                                tmp += [dic[key] for key in dic.keys()]

                            simDf.loc[len(simDf.index)] = tmp
                        else:
                            sims = analysis.multi_analysis(
                                rep1,
                                rep2,
                                preprocFuns,
                                simFuns,
                                names=analysisNames,
                            )
                            simDf.loc[len(simDf.index)] = [v] + [
                                sims[fun] for fun in sims.keys()
                            ]

                    # Save
                    simDf.to_csv(outPath, index=False)
                else:
                    print(f"{outPath} already exists, skipping.", flush=True)
        elif args.analysis == "maxAug":
            for layer in args.layer_index:
                print(f"Working on layer {layer}.", flush=True)
                # Get transforms generators
                transforms = yield_transforms(
                    args.analysis,
                    model,
                    int(layer),
                    dataset,
                    return_aug=False,
                    versions=args.versions,
                    slice=args.version_slice,
                )

                # Create dataframe
                simDf = pd.DataFrame(columns=["version"] + analysisNames)

                outPath = os.path.join(
                    basePath,
                    f"{modelName.split('.')[0]}l{layer}-{args.analysis}{'-v'+str(args.version_slice) if args.version_slice is not None else ''}.csv",
                )

                if not os.path.exists(outPath):
                    # Get similarity measure per transform
                    for v, rep1, rep2 in transforms:
                        # Calculate similarity for each direction
                        simDirs = []
                        for rep in rep2:
                            rep = np.array(rep)
                            simDirs += [
                                analysis.multi_analysis(
                                    rep1, rep, preprocFuns, simFuns, analysisNames
                                )
                            ]

                        # Save the minimum similarity
                        for fun in analysisNames:
                            simDf.loc[len(simDf.index)] = [
                                5,
                                min([dic[fun] for dic in simDirs]),
                            ]

                    # Save
                    simDf.to_csv(outPath, index=False)
                else:
                    print(f"{outPath} already exists, skipping.", flush=True)
        elif args.analysis == "random":
            for layer in args.layer_index:
                print(f"Working on layer {layer}.", flush=True)
                # Get transforms generators
                transforms = yield_transforms(
                    args.analysis,
                    model,
                    int(layer),
                    dataset,
                    return_aug=False,
                    versions=args.versions,
                )

                # Create dataframe
                simDf = pd.DataFrame(columns=["repeat"] + analysisNames)

                outPath = os.path.join(
                    basePath,
                    f"{modelName.split('.')[0]}l{layer}-{args.analysis}{'-repeat'+str(args.version_slice) if args.version_slice is not None else ''}.csv",
                )

                if not os.path.exists(outPath):
                    # Get similarity measure per transform
                    for v, rep1, rep2 in transforms:
                        # Calculate similarity for each direction
                        sims = analysis.multi_analysis(
                            rep1, rep2, preprocFuns, simFuns, analysisNames
                        )

                        # Save similarity into dataframe
                        simDf = pd.concat(
                            [
                                simDf,
                                pd.DataFrame(
                                    [[v] + [sims[fun] for fun in sims.keys()]],
                                    columns=["repeat"] + analysisNames,
                                ),
                            ]
                        )

                    # Save
                    simDf.to_csv(outPath, index=False)
                else:
                    print(f"{outPath} already exists, skipping.", flush=True)
        elif args.analysis == "accuracy":
            augList = ["zoom", "reflect", "color", "noise", "translate"]
            dataLabels = np.load(args.labels_file)

            # First handle augment tests first
            for aug in augList:
                # Make transforms, note selecting first layer for efficiency sake
                transforms = yield_transforms(
                    aug,
                    model,
                    1,
                    dataset,
                    return_aug=True,
                    slice=args.version_slice,
                )

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
                                direct, dataLabels, batch_size=128
                            )

                        accDF.loc[len(accDF.index)] = [float(v.numpy())] + accs
                    else:
                        _, acc = model.evaluate(transImg, dataLabels, batch_size=128)
                        accDF.loc[len(accDF.index)] = [v, acc]

                # Save
                outPath = os.path.join(
                    basePath,
                    f"{modelName[0:-3]}-acc-{aug}{'-v'+str(args.version_slice) if args.version_slice is not None else ''}.csv",
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
                _, acc = dropModel.evaluate(dataset, dataLabels, batch_size=128)
                accDF.loc[len(accDF.index)] = [drop, acc]

            outPath = os.path.join(basePath, f"{modelName[0:-3]}-acc-drop.csv")
            accDF.to_csv(outPath, index=False)
        elif args.analysis == "dropout":
            layerIdx = [int(idx) for idx in args.layer_index]
            layerIdx.sort()
            dropRates = np.arange(0, 1, 0.05)

            data = pd.DataFrame(columns=analysisNames + ["layer", "dropRate"])
            for drop in dropRates:
                if args.version_slice is not None and not args.version_slice == drop:
                    continue
                dropModel = make_dropout_model(model, layerIdx, drop)
                rep1 = dropModel.predict(dataset)
                rep2 = dropModel.predict(dataset)

                # Make sure it's a list
                rep1 = rep1 if isinstance(rep1, list) else [rep1]
                rep2 = rep2 if isinstance(rep1, list) else [rep2]

                for i in range(len(rep1)):
                    sims = analysis.multi_analysis(
                        rep1[i], rep2[i], preprocFuns, simFuns, names=analysisNames
                    )
                    sims["layer"] = layerIdx[i]
                    sims["dropRate"] = drop
                    data.loc[len(data.index)] = [sims[fun] for fun in sims.keys()]

            outPath = os.path.join(
                basePath,
                f"{modelName.split('.')[0]}-dropout-{','.join([str(idx) for idx in layerIdx])}.csv",
            )
            data.to_csv(outPath, index=False)
    else:  # Main
        model = tf.keras.applications.vgg16.VGG16()
        dataset = np.load("../outputs/masterOutput/bigDatasetTestVGG.npy")
        preprocFuns, simFuns, analysisNames = analysis.get_funcs("eucRsa")
        yield_big_transforms("maxAug", model, lambda x: x, 9, dataset)
        simList = []
        for v in range(6):
            transforms = yield_big_transforms("maxAug", model, lambda x: x, 9, dataset)

            for _, rep1, rep2 in transforms:
                simDirs = []
                for rep in rep2:
                    rep = np.array(rep)
                    simDirs += [
                        analysis.multi_analysis(
                            rep1, rep, preprocFuns, simFuns, analysisNames
                        )
                    ]
                sims = [list(sim.values())[0] for sim in simDirs]

                simList.append(max(sims))

        simList
