# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 14:43:01 2019

@author: TMU
"""

import re
import fasttext

out_file = 'essential.cv.ft.txt'
fout = open('eg.ind.ft.txt','w')

# Generating fastText format files
spe_character = []

fpos = 'essential.fasta'
with open(fpos) as f:
    for line in f:
        if (line.startswith('>') == False) and (any(x in line for x in spe_character) == False):
            sequence = ''.join(line).replace('\n','')
            print(sequence)
            fout.write('__label__pos ' + ' '.join(list(sequence)) + '\n')

           
fneg = 'non_essential.fasta'
with open(fneg) as f:
    for line in f:
        if len(line.strip()) > 0:
            if (line.startswith('>') == False) and (any(x in line for x in spe_character) == False):
                sequence = ''.join(line).replace('\n','')
                print(sequence)
                fout.write('__label__neg ' + ' '.join(list(sequence)) + '\n')
            
fout.close()

# Training fastText model
model = fasttext.train_supervised(input=out_file, lr=1.0, epoch=25, wordNgrams=6)

# Getting vectors
model.get_sentence_vector(out_file)