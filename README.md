#An image compression classifier and dataset

The aim is to detect whether an image (patch?) is compressed once (comp1), more than once (comp2) or not at all (uncompressed).

createDataset.py takes a folder (with sub-folders) of images as input and produces a labelled dataset. Commented out code in the file will let you create the comp2 class.

patchRGB.py will take a folder of images and turn it into patches. The size of the patches and how much they overlap (or don't!) depends on the arguments given to patchRGB.py.

trainModel.py will train a CNN on a folder of images. Various architectures and various hyperparameters are available, and the program is designed to be run and ignored while it trains a lot of networks and then returns results telling you which ones are the best architectures in terms of accuracy.
