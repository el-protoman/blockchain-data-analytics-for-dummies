import fastai

PATH='directoryOf/imagesToTrain'
arch = resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch,sz))
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(0.01, 3)