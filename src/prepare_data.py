"""
Kourosh - Prepare Training Corpus
==================================
Writes a Persian training corpus into data/raw/corpus.txt.
The embedded sample is classic public-domain literature (Saadi, Hafez) —
enough to demonstrate learning. For better results, append your own text
to data/raw/corpus.txt (news, books, Wikipedia dumps, etc.).
"""
from src.config import RAW_DATA_DIR


# ---------- Built-in demo corpus (public domain, ~3 KB) ----------
CORPUS = """منت خدای را عز و جل که طاعتش موجب قربت است و به شکر اندرش مزید نعمت. هر نفسی که فرو می‌رود ممد حیات است و چون برمی‌آید مفرح ذات. پس در هر نفس دو نعمت موجود است و بر هر نعمت شکری واجب.

از دست و زبان که برآید کز عهده شکرش به درآید

باری تعالی شکر نعمت خود را بر ما تمام کرد و دست طمع ما از دامن کرمش کوتاه. یکی از ملوک را گفتند فلان کس در بیابان فرورفت. گفت اگر همتش بلند است به منزل خود خواهد رسید.

هر که را طالع فراز آید از چاه به کوه فرازآید. و آن که را طالع فرود آید از کوه به چاه فرود آید.

پادشاهی را شنیدم که به کشتن اسیری اشارت کرد. آن بیچاره در آن حالت ناامیدی، ملک را دشنام دادن گرفت و سقط گفتن. پادشاه پرسید که چه می‌گوید. یکی از وزرا گفت ای خداوند می‌گوید که خوبی و نیکویی با مستحقان. ملک گفت عجب دروغی که با آنچه من گمان دارم موافق است.

یکی را از ملوک گفتند فلان کس را دشنام داد. گفت اگر دشنام من گوید من دشنام او گویم. یکی از حکما را پرسیدند از معنی این سخن. گفت ما ظننت به راستی که دوستی دارم که دشمن من باشد.

حکایت: یکی از ملوک را شنیدم که به کشتن اسیری اشارت کرد. اسیر گفت ای خداوند اگر مرا ببخشی، شکرگزار باشم. ملک گفت تو را بخشیدم که این سخن پسندیده بود.

دوستی را دیدم که درویشی پیشه کرده بود و گوشه نشینی. یکی از ملوک او را گفت اگر خواهی که از خواص ما باشی. گفت اگر خواهم که از خواص تو باشم، از خاصیت خود بیرون آیم.

ای بسا کس که در لباس درویشی پادشاه است و ای بسا پادشاه که در لباس توانگری گدای. یکی را پرسیدند که درویشی چیست. گفت آنکه در دل غنی باشد و در تن رنجور.

به جهان خرم از آنم که جهان خرم از اوست
عاشقم بر همه عالم که همه عالم از اوست

دلا بسوز که سوز تو کارها بکند
نیاز نیم شبی دفع صد بلا بکند

رسید مژده که ایام غم نخواهد ماند
چنان نماند چنین نیز هم نخواهد ماند

هر آن که جانب اهل خدا نگه دارد
خداش در همه حالی به دین خود دارد
"""


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / "corpus.txt"

    if out_path.exists():
        print(f"Corpus already exists: {out_path}")
        print("To overwrite, delete the file first. To append, edit it directly.")
    else:
        out_path.write_text(CORPUS, encoding="utf-8")
        print(f"Wrote corpus: {out_path}")
        print(f"Size: {len(CORPUS):,} characters")

    print("\nTip: for better generation, append your own Persian text to this file")
    print("     (news, books, poetry, Wikipedia, etc.). More data = smarter Kourosh.")


if __name__ == "__main__":
    main()