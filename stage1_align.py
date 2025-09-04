from constant import *


def optimal_align_via_dp(matrix):
    if not matrix or not matrix[0]:
        return 0, []

    m = len(matrix)
    n = len(matrix[0])

    prev_dp = [-float('inf')] * m
    prev_dp[0] = matrix[0][0]
    predecessor = [[-1] * m for _ in range(n)]

    for y in range(1, n):
        curr_dp = [-float('inf')] * m
        current_predecessor = [-1] * m

        max_prev = prev_dp[0]
        curr_dp[0] = matrix[0][y] + max_prev
        current_predecessor[0] = 0

        max_x = 0
        for x in range(1, m):
            if prev_dp[x] > max_prev:
                max_prev = prev_dp[x]
                max_x = x
            curr_dp[x] = matrix[x][y] + max_prev
            current_predecessor[x] = max_x

        prev_dp = curr_dp
        predecessor[y] = current_predecessor

    path = []
    current_x, current_y = m - 1, n - 1
    path.append((current_x, current_y))

    while current_y > 0:
        current_x = predecessor[current_y][current_x]
        current_y -= 1
        path.append((current_x, current_y))

    path.reverse()
    return prev_dp[m - 1], path


def get_alignment(src_lines, tgt_lines, kiwi_model):
    data = []
    for src in src_lines:
        for tgt in tgt_lines:
            data.append(
                {
                    "src": src.strip(),
                    "mt": tgt.strip(),
                }
            )
    all_s = kiwi_model.predict(data, batch_size=EVAL_BS, gpus=1)[0]

    src_num = len(src_lines)
    tgt_num = len(tgt_lines)

    matrix_s = []
    for idx in range(src_num):
        tmp_s = all_s[idx * tgt_num: (idx + 1) * tgt_num]
        matrix_s.append(tmp_s)

    return matrix_s


def restruct_tgt_lines(src_lines, tgt_lines, src_lang, tgt_lang, kiwi_model):
    src_num = len(src_lines)
    tgt_num = len(tgt_lines)

    matrix_s = get_alignment(src_lines, tgt_lines, kiwi_model)

    _, src2tgt_map_list = optimal_align_via_dp(matrix_s)

    new_tgt_lines = []
    for idx in range(src_num):
        hit_idx_list = [_[1] for _ in src2tgt_map_list if _[0] == idx]
        hit_tgt_lines = [tgt_lines[_] for _ in hit_idx_list]
        new_tgt_lines.append(PUNT_MERGE_MAP[tgt_lang].join(hit_tgt_lines).strip())
    return new_tgt_lines
