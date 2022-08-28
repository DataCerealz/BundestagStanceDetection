import os.path

import pandas as pd
import nltk

from data_prep import get_bundestag_df
from textblob_de import TextBlobDE as TextBlob
from deep_translator import GoogleTranslator, LibreTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandarallel import pandarallel  # parallelization
from tqdm import tqdm

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_polarity(input_speech):
    speech = TextBlob(input_speech)
    return speech.sentiment.polarity


# translate text to english because vader is only working properly in english language
def translate_en(text):
    translated = GoogleTranslator(source='de', target='en').translate(text)
    # translated = LibreTranslator(source='de', target='en', base_url = '127.0.0.1:4000').translate(text=text)
    return translated


def translate_speech(text):
    # PROBLEM: translation is limited TO 5000 characters!!!
    # split text after 4999 characaters, translate both path separatly and join them afterwards
    if len(text) > 5000:
        cut_text = text[:4999]
        for idx, c in enumerate(reversed(cut_text)):
            # print(idx, c)
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


def translate_speech_new(text):
    # PROBLEM: translation is limited TO 5000 characters!!!
    # split text after 4999 characaters, translate both path separatly and join them afterwards
    sentences = nltk.sent_tokenize(text)  # this gives us a list of sentences

    all_paragraphs = [""]
    paragraph_index = 0

    for sentence in sentences:
        if len(all_paragraphs[paragraph_index]) > 4500:
            all_paragraphs[paragraph_index] += sentence
        else:
            paragraph_index += 1
            all_paragraphs.append("")
            all_paragraphs[paragraph_index] += sentence

    translated_text = ""

    for paragraph in all_paragraphs:
        translated_text += translate_en(paragraph)

    return translated_text



def get_polarity_vader(input_speech):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(input_speech)
    return sentiment_dict["compound"]


if __name__ == "__main__":
    tqdm.pandas()

    print("Fetching base data...")

    if not os.path.exists('Bundestagsdaten.csv'):
        print("Regenerating dataset from raw files...")
        df = get_bundestag_df(['../Plenarprotokolle_Wahlperiode_19', '../Plenarprotokolle_Wahlperiode_20'])
        df.to_csv("Bundestagsdaten.csv", sep=';')
    else:
        reload_dataset = input("Cached data found. Use cached table instead of regenerating? (Y/N)")

        if not type(reload_dataset) == str:
            print('Invalid input. Exiting.')
            exit()
        else:
            if reload_dataset.lower() == 'y':
                print("Using cached dataset...")
                df = pd.read_csv("Bundestagsdaten.csv", sep=';')
            elif reload_dataset.lower() == 'n':
                df = get_bundestag_df(['../Plenarprotokolle_Wahlperiode_19', '../Plenarprotokolle_Wahlperiode_20'])
                df.to_csv("Bundestagsdaten.csv", sep=';')
            else:
                print('Invalid input. Exiting.')
                exit()



    print("Starting analysis...")

    # run_in_parallel = input('Run textblob analysis in parallel? (Y/N)')
    #
    # if run_in_parallel == 'Y':
    #     try:
    #         pandarallel.initialize()
    #         df["polarity_textblob"] = df.parallel_apply(lambda row: get_polarity(row["speech"]), axis=1)
    #     except Exception as e:
    #         print(f"{bcolors.WARNING}Parallel run failed. Following exception was raised: {e}. Running at default single threat.{bcolors.ENDC}")
    #         df["polarity_textblob"] = df.progress_apply(lambda row: get_polarity(row["speech"]), axis=1)
    #         print("Textblob analysis done.")
    # else:
    #     print("Running Textblob analysis in single threat...")
    #     df["polarity_textblob"] = df.progress_apply(lambda row: get_polarity(row["speech"]), axis=1)
    #     print("Textblob analysis done.")

    try:
        print("Translating to english for vader analysis...")
        df["translated_speech"] = df.progress_apply(lambda row: translate_speech_new(row["speech"]), axis=1) # translate_speech
        print("Translation done.")
    except Exception as e:
        print(f"{bcolors.FAIL}Translation failed. Following exception was raised: {e}.{bcolors.ENDC}")

    try:
        print("Running VADER analysis...")
        df["polarity_vader"] = df.progress_apply(lambda row: get_polarity_vader(row["translated_speech"][:4999]), axis=1)
        print("Vader analysis done.")
    except Exception as e:
        print(f"{bcolors.FAIL}Vader analysis failed. Following exception was raised: {e}.{bcolors.ENDC}")

    try:
        print("Writing results to results.csv...")
        df.to_csv("result.csv", sep=';')
        print("Success. Exiting.")
    except Exception as e:
        print(f"{bcolors.FAIL}Saving results failed. Following exception was raised: {e}. Exiting.{bcolors.ENDC}")
