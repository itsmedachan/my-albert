import sentencepiece as spm
spm.SentencePieceTrainer.Train('--pad_id=0, --unk_id=1, --bos_id=2, --eos_id=3, --pad_piece=<pad>, --unk_piece=<unk>, --bos_piece=<s>, --eos_piece=</s>, --user_defined_symbols=<mask>, --input=jawiki_removed_doc_tag.txt, --model_prefix=jawiki --character_coverage=0.9995 --vocab_size=16000', --input_sentence_size=1000000, --shuffle_input_sentence=true')
