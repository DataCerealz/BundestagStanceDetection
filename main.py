from data_prep import get_bundestag_df
from textblob_de import TextBlobDE as TextBlob
from deep_translator import GoogleTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandarallel import pandarallel  # parallelization
pandarallel.initialize()


def get_polarity(input_speech):
    speech = TextBlob(input_speech)
    return speech.sentiment.polarity


#translate text to english because vader is only working properly in english language
def translate_en(text):
    translated = GoogleTranslator(source='auto', target='en').translate(text)
    return translated

def translate_speech(text):
    # PROBLEM: translation is limited TO 5000 characters!!!
    #split text after 4999 characaters, translate both path separatly and join them afterwards
    if len(text) > 5000:
        cut_text= text[:4999]
        for idx, c in enumerate(reversed(cut_text)):
            #print(idx, c)
            if c == " ":
                print(idx, c)
                first_part = text[:4999 - idx]
                first_part_trans = translate_en(first_part)
                second_part = text[len(first_part):]
                second_part_trans = translate_en(second_part)
                combined_text = first_part_trans + " " + second_part_trans
                print(combined_text)
                return combined_text
    return translate_en(text)

def get_polarity_vader(input_speech):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(input_speech)
    return sentiment_dict["compound"]


if __name__ == "__main__":
    df = get_bundestag_df(['Plenarprotokolle_Wahlperiode_19', 'Plenarprotokolle_Wahlperiode_20'])

    # textblob german
    df["polarity_textblob"] = df.parallel_apply(lambda row: get_polarity(row["speech"]), axis=1)

    df["translated_speech"] = df.apply(lambda row: translate_speech(row["speech"]), axis=1)

    df["polarity_vader"] = df.apply(lambda row: get_polarity_vader(row["translated_speech"][:4999]), axis=1)

    df.to_csv("result.csv", sep=';')

