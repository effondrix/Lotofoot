import urllib.request
import re
import numpy as np
import pandas as pd
import sys
import datetime
import difflib
from unidecode import unidecode
from bs4 import BeautifulSoup


import random

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

PAGE_NUMBER_POSITION_IN_URL = 3
DAY_INDEX = 0
TOPIC_URL_INDEX = 1
END_TIME_INDEX = 1
INCLUDE_QUOTES = True
DELETED_PSEUDO = "Pseudo supprimé"

DAY = 'Journée'
END_TIME = 'Date fin'
SCORES = 'Résultats'

months_list_fr = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
                  "novembre", "décembre"]
months_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

teams_dict = {"ASCO": ["Angers SCO", "SCO Angers", "SCO", "Angers", "ASCO"],
              "ASM": ["AS Monaco", "Monaco", "ASM"],
              "ASSE": ["AS Saint-Etienne", "Saint-Etienne", "Etienne", "ASSE"],
              "CF63": ["Clermont Foot 63", "Clermont", "Clermont Foot", "CF63"],
              "DFCO": ["Dijon FCO", "Dijon", "DFCO"],
              "ESTAC": ["ESTAC Troyes", "Troyes", "ESTAC"],
              "FCGB": ["Girondins de Bordeaux", "Girondins", "Bordeaux", "FCGB"],
              "FCL": ["FC Lorient", "Lorient", "FCL"],
              "FCM": ["FC Metz", "Metz", "FCM"],
              "FCN": ["FC Nantes", "Nantes", "FCN"],
              "LOSC": ["Lille OSC", "Lille", "LOSC"],
              "MHSC": ["Montpellier HSC", "Montpellier", "MHSC"],
              "OGCN": ["OGC Nice", "Nice", "OGCN"],
              "OL": ["Olympique Lyonnais", "Lyon", "Lyonnais", "OL"],
              "OM": ["Olympique de Marseille", "Marseille", "OM"],
              "PSG": ["Paris Saint-Germain", "Paris", "PSG"],
              "RCL": ["RC Lens", "Lens", "RCL"],
              "RCS": ["RC Strasbourg", "Strasbourg", "RCS"],
              "SB29": ["Stade Brestois", "Brest", "Brestois", "SB29"],
              "SDR": ["Stade de Reims", "Reims", "SDR"],
              "SR": ["Stade Rennais", "Rennes", "Rennais", "SR"]}

scores_list = {"1", "N", "2", "1N", "12", "N2"}

topic_list_path = "topic_list_2021.txt"
topic_list_file = open(topic_list_path, "r")
topic_list = topic_list_file.read().splitlines()

resultats_path = "resultats_2021.txt"


def format_date(date_str):
    date_splitted = date_str.split(" ")
    date_splitted.pop(3)
    month = date_splitted[1]
    month = months_list[months_list_fr.index(month)]
    date_us = " ".join([month, date_splitted[0], date_splitted[2], date_splitted[3]])
    return datetime.datetime.strptime(date_us, '%b %d %Y %H:%M:%S')


def next_page(url):
    splitted_url = url.split(separator)
    splitted_url[PAGE_NUMBER_POSITION_IN_URL] = str(int(splitted_url[PAGE_NUMBER_POSITION_IN_URL]) + 1)
    return separator.join(splitted_url)


def find_team_in_word(teams_dict, word):
    for team_acronym, team_synonyms in teams_dict.items():
        for team_synonym in team_synonyms:
            if difflib.SequenceMatcher(None, team_synonym.lower(), word.lower()).ratio() > 0.9:
                return team_acronym
    return None


def get_team_acronym(teams_dict, word):
    for team_acronym, team_synonyms in teams_dict.items():
        for team_synonym in team_synonyms:
            if difflib.SequenceMatcher(None, team_synonym.lower(), word.lower()).ratio() > 0.9:
                return team_acronym
    assert False


df_header = [[], [], []]

with open(resultats_path, "r", encoding="utf-8") as resultats_file:
    for resultats_line in resultats_file:
        if resultats_line.startswith("J"):
            resultats_line = resultats_line.rstrip().split(',')
            day_str = resultats_line[DAY_INDEX]
            day = int(day_str[1:])  # remove "J"
            day_results = []
            for i in range(10):
                match = next(resultats_file, "").strip()
                match = match.split("=")
                score_and_end_time = match[1].split(",")
                score, end_time = score_and_end_time[0], score_and_end_time[1]
                teams = match[0].strip()
                teams = teams.split("/")
                team1, team2 = teams[0].strip(), teams[1].strip()
                day_results.append(
                    [get_team_acronym(teams_dict, unidecode(team1)), get_team_acronym(teams_dict, unidecode(team2)),
                     score,end_time])
                # print(day_results)

            df_header[0].append(day_str)
            df_header[1].append(str(day_results))


tuples = list(zip(df_header[0],df_header[1]))
index = pd.MultiIndex.from_tuples(tuples, names=[DAY, SCORES])
df = pd.DataFrame(columns=index)

df.to_csv(r'lotofoot_2021.csv', index=True, header=True)
print(df)


EMPTY_ROW = np.full((1, len(df.columns)), "[0,0,0,0,0,0,0,0,0,0]")[0]
pronostics = []

separator = '-'

regex_bloc_message = re.compile('.*text-enrichi-forum.*')
regex_bloc_edition = re.compile('.*info-edition-msg.*')
regex_bloc_pseudo = re.compile('.*bloc-pseudo-msg.*')
regex_bloc_contenu = re.compile('.*bloc-contenu.*')
regex_bloc_date = re.compile('.*bloc-date-msg.*')

for topic_list_line in topic_list:
    topic_list_line = topic_list_line.split(',')
    day_str = topic_list_line[DAY_INDEX]
    day = int(day_str[1:])  # remove "J"
    scores = eval(df.columns.get_level_values(SCORES)[day - 1])
    print("New Day:", day_str)
    topic_url = topic_list_line[TOPIC_URL_INDEX]
    page_number = 1
    # print(pronostics)
    print("_" * 200)
    pronostics = []

    page = urllib.request.urlopen(topic_url)

    while topic_url == page.geturl():
        print(day_str + " Page:", page_number)
        print(topic_url)

        html = page.read().decode("utf-8")
        soup = BeautifulSoup(html, "html.parser")

        # print(html)

        pseudo_box = soup.findAll(["div", "span"], attrs={"class": regex_bloc_pseudo}, text=True)
        contenu_box = soup.findAll("div", attrs={"class": regex_bloc_contenu})
        date_box = soup.findAll("div", attrs={"class": regex_bloc_date})

        assert (len(contenu_box) == len(pseudo_box) == len(date_box))

        for pseudo, contenu, date in zip(pseudo_box, contenu_box, date_box):
            pseudo = pseudo.text.strip()
            if pseudo == DELETED_PSEUDO:
                continue
            # print(df)
            # print("__"*100)
            if pseudo not in df.index:
                df.loc[pseudo] = EMPTY_ROW
            date = date.text.strip()
            message_box = contenu.findAll("div", attrs={"class": regex_bloc_message})
            edition_box = contenu.findAll("div", attrs={"class": regex_bloc_edition})

            message_is_edited = (len(edition_box) > 0)
            if message_is_edited:
                date_in_edition_txt = edition_box[0].text.strip()
                edition_date = date_in_edition_txt.split('le ')[1].split(' par')[0]
                date = edition_date

            date = format_date(date)

            assert (len(message_box) == 1)
            quote_box = message_box[0].findAll("blockquote")

            is_quote = (len(quote_box) > 0)

            if is_quote:
                text_in_quote = quote_box[0].text
                text_in_quote = text_in_quote.split("\n")
                first_line_text_in_quote = text_in_quote[0]
                if first_line_text_in_quote.endswith("a écrit :"):
                    is_quote = False

            for tag in message_box[0].select("blockquote"):
                tag.decompose()

            message = message_box[0].get_text(separator = "\n")
            message = message.split("\n")[1:-1]
            if is_quote:
                message = text_in_quote + message

            khey_pronostic = eval(df.loc[pseudo][day-1])

            for message_line in message:
                message_line = unidecode(message_line.split('\n')[0])
                message_line = re.split('[^a-zA-Z0-9]', message_line)
                teams_in_line = []
                score_in_line = []
                for word in message_line:
                    if len(word) == 0: continue
                    if len(word) < 3:
                        if word.upper() in scores_list:
                            score_in_line.append(word.upper())
                    else:
                        team_in_word = find_team_in_word(teams_dict, word)
                        if team_in_word is not None:
                            if team_in_word not in teams_in_line:
                                teams_in_line.append(team_in_word)

                if not (len(teams_in_line) == 2 and  len(score_in_line) == 1 ):
                    continue

                for i in range(len(scores)):
                    if scores[i][0:2] == teams_in_line:
                        end_time = datetime.datetime.strptime(scores[i][-1].strip(), '%b %d %Y %H:%M:%S')
                        if date < end_time:
                            khey_pronostic[i] = score_in_line[0]

            df.at[pseudo, day_str] = str(khey_pronostic)

        topic_url = next_page(topic_url)
        page = urllib.request.urlopen(topic_url)

        page_number += 1


df.to_csv(r'lotofoot_2021.csv', index=True, header=True)