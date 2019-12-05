import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("m.model")
print(sp.EncodeAsIds