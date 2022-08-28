import os

import pandas as pd
from bs4 import BeautifulSoup


def parse_data_to_df(file):
    xml_data = open(file, 'r').read()  # Read file
    soup = BeautifulSoup(xml_data, 'xml')

    combinend_tgsordpkt = []
    combinend_speaker = []
    combinend_party = []
    combinend_speech = []

    # get the date of the Plenarsitzung
    date = [soup.find("datum").attrs["date"]]

    tagesordnungspunkt = soup.find_all('tagesordnungspunkt')
    for punkt in tagesordnungspunkt:

        # ignore tagesordnungspunkt if there is no redner
        check_for_redner = len(punkt.find_all("rede")) != 0
        if check_for_redner:

            tgsordpkt_all_text = []  # list to store the text from the speeches joined as one string per speaker
            tgsordpkt_all_speaker = []  # list to store the text from the speeches joined as one string per speaker
            tgsordpkt_thema = []
            tgsordpkt_speaker_party = []

            thema = punkt.find('p', {"klasse": "T_fett"})
            if thema is None:
                continue
            thema_txt = thema.get_text()
            tgsordpkt_thema.append(thema_txt)
            reden = punkt.find_all('rede')
            for rede in reden:
                # extract the text from the speeches
                plain_text = []
                text = rede.find_all('p')
                ignore_tags = ["redner"]
                for txt in text:
                    # print(txt.attrs)
                    if txt.has_attr('klasse'):
                        if txt["klasse"] in ignore_tags:
                            continue
                    plain_text.append(txt.get_text())
                joined_text = ' '.join(plain_text)
                tgsordpkt_all_text.append(joined_text)

                # extract the speaker "redner" of the speech
                redner = rede.find('p', {"klasse": "redner"})
                store_redner_info = []
                for re in redner:
                    store_redner_info.append(re.get_text())

                get_speaker = store_redner_info[-1]
                tgsordpkt_all_speaker.append(get_speaker)

                # extract party from redner
                if "(" in get_speaker:
                    count = 0
                    while True:
                        count = count + 1
                        character = get_speaker[-count]
                        if character == "(":
                            tgsordpkt_speaker_party.append(get_speaker[-count:])
                            break
                else:
                    tgsordpkt_speaker_party.append('N/A')

            if len(tgsordpkt_thema) < len(tgsordpkt_all_speaker):
                while len(tgsordpkt_thema) < len(tgsordpkt_all_speaker):
                    tgsordpkt_thema.append(tgsordpkt_thema[0])

            combinend_tgsordpkt.append(tgsordpkt_thema)
            combinend_speaker.append(tgsordpkt_all_speaker)
            combinend_speech.append(tgsordpkt_all_text)
            combinend_party.append(tgsordpkt_speaker_party)

    combinend_tgsordpkt = [item for sublist in combinend_tgsordpkt for item in sublist]
    combinend_speaker = [item for sublist in combinend_speaker for item in sublist]
    combinend_speech = [item for sublist in combinend_speech for item in sublist]
    combinend_party = [item for sublist in combinend_party for item in sublist]

    if len(date) < len(combinend_tgsordpkt):
        while len(date) < len(combinend_tgsordpkt):
            date.append(date[0])

    test_for_content = combinend_tgsordpkt + combinend_speaker + combinend_speech + combinend_party
    if len(test_for_content) != 0:
        # dictionary of lists
        dict = {
            'date': date,
            'thema': combinend_tgsordpkt,
            'speaker': combinend_speaker,
            'party': combinend_party,
            'speech': combinend_speech
        }

        df = pd.DataFrame(dict)
        return df


def get_bundestag_df(liste_von_plenarprotokollverzeichnissen):
    # assign directory
    directories = liste_von_plenarprotokollverzeichnissen

    # iterate over files in
    # that directory

    dfs = []
    for directory in directories:
        print("Processing directory " + directory)
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):

                get_df = parse_data_to_df(f)
                dfs.append(get_df)
        print("Done processing directory " + directory)

    return pd.concat(dfs, ignore_index=True)
