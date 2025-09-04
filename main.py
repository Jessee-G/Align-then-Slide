from comet import download_model, load_from_checkpoint
from pre_seg import sep_doc
from stage1_align import restruct_tgt_lines
from stage2_slide import n_slide_comet_with_ref, n_slide_comet_no_ref


da20_model = load_from_checkpoint("Unbabel/wmt20-comet-da/checkpoints/model.ckpt")
# da22_model = load_from_checkpoint("Unbabel/wmt22-comet-da/checkpoints/model.ckpt")
kiwi_model = load_from_checkpoint("Unbabel/wmt22-cometkiwi-da/checkpoints/model.ckpt")


if __name__ == "__main__":
    src_lang = "zh"
    tgt_lang = "en"

    src_text = "您正在收看BBC世界新闻头条。今天，将在哈里王子和梅根·马克尔婚礼上穿过温莎城堡的马车队伍要进行彩排。" \
               "有250多名武装部队成员参加，还有成千上万的围观者和世界各地的媒体在关注。欧盟将召开15年以来与巴尔干半岛国家的首次联合峰会。" \
               "此次聚集在保加利亚的核心议题是改善与六个巴尔干半岛国家的关系，它们志在加入此集团。" \
               "刚果民主共和国的埃博拉爆发从乡村地区扩散到了城市，人们担心这种疾病会越来越难控制。" \
               "已知有23人死亡。特朗普总统承认他去年向个人律师迈克尔·科恩偿还了十几万美元。"
    ref_text = "You are watching the BBC World News headlines. " \
               "Today, a rehearsal will take place for the carriage procession " \
               "that will pass through Windsor Castle during Prince Harry and Meghan Markle's wedding. " \
               "Over 250 members of the armed forces will participate, " \
               "with thousands of onlookers and media from around the world in attendance. " \
               "The European Union will hold its first joint summit with Balkan countries in 15 years. " \
               "The main focus of this gathering in Bulgaria is to improve " \
               "relations with the six Balkan countries that aspire to join the bloc. " \
               "The Ebola outbreak in the Democratic Republic of Congo has spread " \
               "from rural areas to urban centers, raising concerns that " \
               "the disease will become increasingly difficult to control. At least 23 deaths have been reported. " \
               "President Trump acknowledged that he repaid his personal lawyer Michael Cohen over $100,000 last year."
    mt_text = "You are watching BBC World News headlines. " \
              "Today, a rehearsal is being held for the procession of carriages " \
              "that will pass through Windsor Castle during Prince Harry and Meghan Markle's wedding. " \
              "More than 250 members of the armed forces are participating, " \
              "along with thousands of onlookers and media from around the world watching closely. " \
              "The European Union will hold its first joint summit with Balkan countries in 15 years. " \
              "The main focus of the gathering in Bulgaria is to improve " \
              "relations with six Balkan countries that are seeking to join the bloc. " \
              "The Ebola outbreak in the Democratic Republic of Congo has spread from rural areas to cities, " \
              "raising concerns that the disease will become increasingly difficult to control. " \
              "There have been 23 confirmed deaths. " \
              "President Trump admitted that he repaid his personal lawyer, " \
              "Michael Cohen, more than a hundred thousand dollars last year."

    src_lines = sep_doc(src_text, src_lang)
    ref_lines = sep_doc(ref_text, tgt_lang)
    mt_lines = sep_doc(mt_text, tgt_lang)

    new_ref_lines = restruct_tgt_lines(src_lines, ref_lines, src_lang, tgt_lang, kiwi_model)
    new_mt_lines = restruct_tgt_lines(src_lines, mt_lines, src_lang, tgt_lang, kiwi_model)

    s1, s2, s3, s4 = n_slide_comet_with_ref(src_lines, new_ref_lines, new_mt_lines, src_lang, tgt_lang, da20_model)
    print("With Ref:", (s1 + s2 + s3 + s4) / 4)

    s1, s2, s3, s4 = n_slide_comet_no_ref(src_lines, new_mt_lines, src_lang, tgt_lang, da20_model)
    print("No Ref:", (s1 + s2 + s3 + s4) / 4)



