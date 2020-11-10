import fasttext

model = fasttext.train_supervised(input="cooking.train", lr=1.0, epoch=25, wordNgrams=6)