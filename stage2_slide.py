from constant import PUNT_MERGE_MAP, EVAL_BS


def n_slide_comet_with_ref(src_lines, ref_lines, tgt_lines, src_lang, tgt_lang, da_model):
    n_gram = [1, 2, 3, 4]
    n_gram_s = []
    for step in n_gram:
        data = []
        for i in range(0, len(src_lines) + 1 - step, 1):
            data.append(
                {
                    "src": PUNT_MERGE_MAP[src_lang].join(src_lines[i: i + step]),
                    "ref": PUNT_MERGE_MAP[tgt_lang].join(ref_lines[i: i + step]),
                    "mt": PUNT_MERGE_MAP[tgt_lang].join(tgt_lines[i: i + step])
                }
            )
        matrix_s = da_model.predict(data, batch_size=EVAL_BS, gpus=1)[1]
        n_gram_s.append(matrix_s)
    return n_gram_s


def n_slide_comet_no_ref(src_lines, tgt_lines, src_lang, tgt_lang, kiwi_model):
    n_gram = [1, 2, 3, 4]
    n_gram_s = []
    for step in n_gram:
        data = []
        for i in range(0, len(src_lines) + 1 - step, 1):
            data.append(
                {
                    "src": PUNT_MERGE_MAP[src_lang].join(src_lines[i: i + step]),
                    "mt": PUNT_MERGE_MAP[tgt_lang].join(tgt_lines[i: i + step])
                }
            )
        matrix_s = kiwi_model.predict(data, batch_size=EVAL_BS, gpus=1)[1]
        n_gram_s.append(matrix_s)
    return n_gram_s

