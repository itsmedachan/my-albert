import sentencepiece as spm


class SentencePieceTokenizer:
  def __init__(self, MODEL_PATH):
    self.sp = spm.SentencePieceProcessor()
    self.sp.Load(MODEL_PATH)

  def text_to_ids(self, input_text):
    return self.sp.EncodeAsIds(input_text)

  def ids_to_text(self, input_ids):
    return self.sp.DecodeIds(input_ids)

  def text_to_tokens(self, input_text):
    return self.sp.EncodeAsPieces(input_text)
