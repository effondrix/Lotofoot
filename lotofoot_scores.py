import sys
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

from matplotlib.cbook import get_sample_data
from adjustText import adjust_text
from matplotlib import gridspec
import math

csv_headers = ["Journée", "Résultats"]
days = []
scores = []

dirname = os.path.dirname(__file__)
res_dir = os.path.join(dirname, 'res/')

aliases = {"MagnumWhooperFF": ["MagnumWhooperMW", "MagnumWhooperLB", "MagnumWhooper-", "MagnumWhooper92"],
           "Effondrix": ["Effondrix2", "Effondrix3", "Effondrix4"],
           "TaMeufQuiChie4": ["TaMeufQuiChie9"],
           "CafeSucre11": ["CafeSucre12"],
           "Chapeco": ["Chapecoed", "Chapecoense"],
           "Veyzen_I": ["Veyzen_II"],
           "saacri1": ["saacri4"],
           "VValden": ["Waldeden"]}

teams_dict = {"ASCO": ["Angers SCO", "SCO Angers", "SCO", "Angers", "ASCO"],
              "ASM": ["AS Monaco", "Monaco", "ASM"],
              "ASSE": ["AS Saint-Etienne", "Saint-Etienne", "Etienne", "ASSE"],
              "CF63": ["Clermont Foot 63", "Clermont", "Clermont Foot", "CF63"],
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


csv_path = 'lotofoot_2021.csv'

with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            days = row[1:]
            line_count += 1
        elif line_count == 1:
            scores = row[1:]
            line_count += 1
    csv_file.close()

formatted_scores = []

for day_scores in scores:
    scores_current_day = []
    for score in eval(day_scores):
        scores_current_day.append(score[2].strip())
    formatted_scores.append(scores_current_day)


def get_number_of_bets(csv_path):
    with open(csv_path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        number_of_days = len(next(csv_reader)) - 1
        number_of_bets = np.zeros((number_of_days, 10, 3))
        for row in csv_reader:
            pseudo = row[0]
            if pseudo in csv_headers:
                continue
            for i in range(len(row[1:])):
                day_pronostics = eval(row[1:][i])
                for j in range(len(day_pronostics)):
                    match_pronostic = str(day_pronostics[j])
                    if match_pronostic == "1":
                        number_of_bets[i][j][0] += 1
                    elif match_pronostic == "N":
                        number_of_bets[i][j][1] += 1
                    elif match_pronostic == "2":
                        number_of_bets[i][j][2] += 1
                    elif match_pronostic == "1N":
                        number_of_bets[i][j][0] += 1
                        number_of_bets[i][j][1] += 1
                    elif match_pronostic == "N2":
                        number_of_bets[i][j][1] += 1
                        number_of_bets[i][j][2] += 1
                    elif match_pronostic == "12":
                        number_of_bets[i][j][0] += 1
                        number_of_bets[i][j][2] += 1
    return number_of_bets


def get_odds(csv_path):
    number_of_bets = get_number_of_bets(csv_path)
    odds = np.copy(number_of_bets)
    for i in range(len(number_of_bets)):
        for j in range(len(number_of_bets[i])):
            number_of_bets_by_match = number_of_bets[i][j]
            bets_total = sum(number_of_bets_by_match)
            bets_1 = number_of_bets_by_match[0]
            bets_N = number_of_bets_by_match[1]
            bets_2 = number_of_bets_by_match[2]

            if bets_1 == 0:
                odds[i][j][0] = np.nan
            else:
                odds[i][j][0] = round(bets_total / bets_1, 3)
            if bets_N == 0:
                odds[i][j][1] = np.nan
            else:
                odds[i][j][1] = round(bets_total / bets_N, 3)
            if bets_2 == 0:
                odds[i][j][2] = np.nan
            else:
                odds[i][j][2] = round(bets_total / bets_2, 3)
    return odds


def get_principal_pseudo(aliases, pseudo):
    for principal, secondaires in aliases.items():
        for secondaire in secondaires:
            if pseudo == secondaire:
                return principal
    return pseudo


def get_global_ranking(csv_path, day=0):
    pseudos_list = []

    with open(csv_path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            pseudo = row[0]
            if pseudo in csv_headers:
                continue
            pseudo = get_principal_pseudo(aliases, pseudo)
            if pseudo not in [x[0] for x in pseudos_list]:
                pseudos_list.append([pseudo, 0])

            pseudo_position = [index for index, row in enumerate(pseudos_list) if pseudo in row][0]

            if day==0:
                for score, pronostic in zip(formatted_scores, row[1:]):
                    for s, p in zip(score, eval(pronostic)):
                        if p == 0:
                            continue
                        if s == p:
                            pseudos_list[pseudo_position][1] += 1
            else:
                for score, pronostic in zip(formatted_scores[:day], row[1:day+1]):
                    for s, p in zip(score, eval(pronostic)):
                        if p == 0:
                            continue
                        if s == p:
                            pseudos_list[pseudo_position][1] += 1

    return sorted(pseudos_list, key=lambda x: x[1], reverse=True)


def get_score_by_day(csv_path):
    pseudos_list = []
    with open(csv_path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        number_of_days = len(next(csv_reader)) - 1
        for row in csv_reader:
            pseudo = row[0]
            if pseudo in csv_headers:
                continue
            pseudo = get_principal_pseudo(aliases, pseudo)
            if pseudo not in [x[0] for x in pseudos_list]:
                scores_by_day = [0] * number_of_days
                pseudos_list.append([pseudo, scores_by_day])

            pseudo_position = [index for index, row in enumerate(pseudos_list) if pseudo in row][0]

            day = 0
            for score, pronostic in zip(formatted_scores, row[1:]):
                for s, p in zip(score, eval(pronostic)):
                    if p == 0:
                        continue
                    if s == p:
                        pseudos_list[pseudo_position][1][day] += 1
                day += 1
    return sorted(pseudos_list, key=lambda x: sum(x[1]), reverse=True)


def get_cumulative_score(csv_path):
    pseudos_list = get_score_by_day(csv_path)
    for pseudo_index in range(len(pseudos_list)):
        cumulative_score = [0]
        for i in range(len(pseudos_list[pseudo_index][1])):
            cumulative_score.append(pseudos_list[pseudo_index][1][i] + cumulative_score[-1])
        pseudos_list[pseudo_index][1] = cumulative_score
    return pseudos_list


def get_ranking_for_day(csv_path, day):
    pseudos_list = []
    with open(csv_path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            pseudo = row[0]
            if pseudo in csv_headers:
                continue
            pseudo = get_principal_pseudo(aliases, pseudo)
            if pseudo not in [x[0] for x in pseudos_list]:
                day_score = 0
                pseudos_list.append([pseudo, day_score])

            pseudo_position = [index for index, row in enumerate(pseudos_list) if pseudo in row][0]

            score = formatted_scores[day - 1]
            pronostic = row[day]
            for s, p in zip(score, eval(pronostic)):
                if p == 0:
                    continue
                if s == p:
                    pseudos_list[pseudo_position][1] += 1
    return sorted(pseudos_list, key=lambda x: x[1], reverse=True)


def plot_top_players_evolution(csv_path, number_of_players):
    players_cumulative_score = get_cumulative_score(csv_path)
    global_ranking = get_global_ranking(csv_path)
    top_players = []
    pseudos_list = []
    number_of_days = len(players_cumulative_score[0][1])
    x = np.linspace(0.0, number_of_days - 1, number_of_days)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color', plt.cm.rainbow(np.linspace(1, 0, number_of_players)))

    for i in range(number_of_players):
        top_players.append(global_ranking[i][0])
    for player in top_players:
        pseudo_position = [index for index, row in enumerate(players_cumulative_score) if player in row][0]
        player_evolution = players_cumulative_score[pseudo_position][1]
        ax.plot(x, player_evolution)
        pseudos_list.append(players_cumulative_score[pseudo_position][0])

    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(pseudos_list)
    ax.legend(pseudos_list, title='Top ' + str(number_of_players), bbox_to_anchor=(1, 1), loc='upper left', prop=fontP)
    plt.gca().set(ylabel='Points', xlabel='Journée')
    plt.savefig("figures/top_" + str(number_of_players) + "_evolution.png", bbox_inches='tight')
    plt.show()


def plot_ranking(csv_path, number_of_players=50, plot_text_values=False):
    global_ranking = get_global_ranking(csv_path)
    # Draw plot

    pseudos = [player[0] for player in global_ranking][:number_of_players][::-1]
    scores = [player[1] for player in global_ranking][:number_of_players][::-1]
    plt.figure(figsize=(14, 10))

    colors_sampled = plt.cm.CMRmap(np.linspace(1, 0, 256))
    max_score = scores[-1]
    ratio = 255 / max_score
    colors = [colors_sampled[int(s * ratio)] for s in scores]

    plt.hlines(y=pseudos, xmin=0, xmax=scores, colors=colors, alpha=0.8, linewidth=5)

    if plot_text_values:
        tex_list = []
        for x, y, tex in zip(reversed(scores), reversed(pseudos), reversed(scores)):
            if tex not in tex_list:
                tex_list.append(tex)
                plt.text(x, y, "  " + str(round(tex, 2)), horizontalalignment='left',
                         verticalalignment='center', fontdict={'color': 'black', 'size': 10})

    plt.gca().set(ylabel='$Player$', xlabel='$Points$')
    plt.yticks(pseudos, pseudos, fontsize=12)
    plt.title('Top ' + str(number_of_players), fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.4)
    plt.savefig("figures/global_ranking_" + str(number_of_players) + ".png", bbox_inches='tight')
    plt.show()


def plot_ranking_stacked(csv_path, number_of_players=50):
    cumulative_score = get_cumulative_score(csv_path)

    pseudos = [player[0] for player in cumulative_score][:number_of_players][::-1]
    plt.figure(figsize=(14, 10))

    number_of_days = len(cumulative_score[0][1])

    colors_sampled = plt.cm.rainbow(np.linspace(1, 0, number_of_days))

    null_vector = [0] * number_of_players
    if (number_of_days>2):
        for i in range(1, number_of_days):
            day_scores_current = [player[1][-i] for player in cumulative_score][:number_of_players][::-1]
            plt.barh(pseudos, day_scores_current, left=null_vector, color=colors_sampled[i])

    plt.gca().set(ylabel='Player', xlabel='Points')
    plt.title('Top ' + str(number_of_players) + " journée par journée", fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.4)
    plt.savefig("figures/global_ranking_stacked_" + str(number_of_players) + ".png", bbox_inches='tight')
    plt.show()


def plot_score_depending_on_pronostic_number(csv_path, n=1):
    pseudos_with_at_least_n_pronostics = get_pseudos_with_at_least_n_pronostics(csv_path, n)

    pseudos_list = [x[0] for x in pseudos_with_at_least_n_pronostics]
    number_of_pronostics = [x[1] for x in pseudos_with_at_least_n_pronostics]

    pseudos_absolute = [player[0] for player in global_ranking if player[0] in pseudos_list][::-1]
    scores_absolute = [player[1] for player in global_ranking if player[0] in pseudos_list][::-1]

    pseudos_score_and_number_of_pronostics = [[p, s, number_of_pronostics[pseudos_list.index(p)]] for p, s in
                                              zip(pseudos_absolute, scores_absolute)]

    plt.figure(figsize=(17, 11))
    pseudos = [x[0] for x in pseudos_score_and_number_of_pronostics]
    scores = [x[1] for x in pseudos_score_and_number_of_pronostics]
    number_of_pronostics = [x[2] for x in pseudos_score_and_number_of_pronostics]

    margin_ratio = 1.1

    xlim = int(margin_ratio * number_of_pronostics[-1])
    ylim = int(margin_ratio * scores_absolute[-1])

    x = np.linspace(0, xlim, 2)

    colors_sampled = plt.cm.RdYlGn(np.linspace(1, 0, 256))
    for i, q in enumerate(np.linspace(1, 0, 256)):
        plt.plot(x, q * x, color=colors_sampled[i], linewidth=10, zorder=1)

    cst_lines = np.linspace(0, [margin_ratio * x for x in number_of_pronostics], 2)

    for k in np.linspace(0, 1, 11):
        plt.plot(cst_lines, k * cst_lines, color="grey", linestyle='--', dashes=(5, 20), linewidth=0.3, zorder=2,
                 alpha=0.3)
        y_text = int(k * margin_ratio * number_of_pronostics[-1])
        if y_text < ylim:
            plt.text(xlim + 1, y_text, str(int(k * 100)) + "%", fontsize=8)

    plt.scatter(number_of_pronostics, scores, zorder=3)

    plt.xlim(0, xlim)
    plt.ylim(0, ylim)
    texts = []
    for i, txt in enumerate(pseudos):
        texts.append(plt.text(number_of_pronostics[i], scores[i], txt, fontsize=8))
    adjust_text(texts, only_move={'points': 'y', 'texts': 'y'})

    plt.gca().set(xlabel='Nombre de pronostics', ylabel='Score')
    plt.title('Score en fonction du nombre de pronostics', fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.2)

    plt.savefig("figures/score_depending_on_pronostics_number.png", bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(17, 11))
    pseudos = [x[0] for x in pseudos_score_and_number_of_pronostics]
    scores = [x[1] for x in pseudos_score_and_number_of_pronostics]
    number_of_pronostics = [x[2] for x in pseudos_score_and_number_of_pronostics]

    scores_centered = [s - n / 2 for s, n in zip(scores, number_of_pronostics)]

    max_abs_score = abs(max(scores_centered, key=abs))

    margin_ratio = 1.1

    xlim = int(margin_ratio * number_of_pronostics[-1])
    ylim = int(margin_ratio * max_abs_score)

    x = np.linspace(0, xlim, 2)

    colors_sampled = plt.cm.RdYlGn(np.linspace(0, 1, 256))
    for i, q in enumerate(np.linspace(-0.5, 0.5, 256)):
        plt.plot(x, q * x, color=colors_sampled[i], linewidth=10, zorder=1)

    cst_lines = np.linspace(0, [margin_ratio * x for x in number_of_pronostics], 2)

    for k in np.linspace(-0.5, 0.5, 11):
        plt.plot(cst_lines, k * cst_lines, color="grey", linestyle='--', dashes=(5, 20), linewidth=0.3, zorder=2,
                 alpha=0.3)
        y_text = int(k * (margin_ratio - ((margin_ratio - 1) / 2)) * xlim)
        if y_text < ylim and y_text > -ylim:
            plt.text(xlim + 1, y_text, str(int((k + 0.5) * 100)) + "%", fontsize=8)

    for k in np.linspace(-0.2, 0.2, 21):
        plt.plot(cst_lines, k * cst_lines, color="grey", linestyle='--', dashes=(3, 20), linewidth=0.2, zorder=2,
                 alpha=0.1)

    plt.scatter(number_of_pronostics, scores_centered, zorder=3)

    plt.xlim(0, xlim)
    plt.ylim(-ylim, ylim)
    texts = []
    for i, txt in enumerate(pseudos):
        texts.append(plt.text(number_of_pronostics[i], scores_centered[i], txt, fontsize=8))
    adjust_text(texts, only_move={'points': 'y', 'texts': 'y'})

    plt.gca().set(xlabel='$Nombre de pronostics$', ylabel='$Pourcentage de réussite$')
    plt.yticks([])
    plt.title('Pourcentage de réussite en fonction du nombre de pronostics', fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.2)

    plt.savefig("figures/score_depending_on_pronostics_number_2.png", bbox_inches='tight')
    plt.show()


def plot_bets_global_ranking(csv_path, n=1, plot_text_values=False):
    bets_global_ranking = get_bets_global_ranking(csv_path)

    pseudos_with_at_least_n_pronostics = get_pseudos_with_at_least_n_pronostics(csv_path, n)

    pseudos_list = [x[0] for x in pseudos_with_at_least_n_pronostics]
    number_of_pronostics = [x[1] for x in pseudos_with_at_least_n_pronostics]

    pseudos_absolute = [player[0] for player in bets_global_ranking if player[0] in pseudos_list][::-1]
    scores_absolute = [player[1] for player in bets_global_ranking if player[0] in pseudos_list][::-1]

    scores_relative = [s / number_of_pronostics[pseudos_list.index(p)] for p, s in
                       zip(pseudos_absolute, scores_absolute)]

    pseudos_score_and_number_of_pronostics = [[p, s, number_of_pronostics[pseudos_list.index(p)]] for p, s in
                                              zip(pseudos_absolute, scores_absolute)]

    pseudos_and_scores_relative = [[p, s] for p, s in zip(pseudos_absolute, scores_relative)]
    pseudos_and_scores_relative = sorted(pseudos_and_scores_relative, key=lambda x: x[1], reverse=False)

    pseudos_relative = [x[0] for x in pseudos_and_scores_relative]
    scores_relative = [x[1] for x in pseudos_and_scores_relative]

    number_of_players = len(pseudos_absolute)
    plt.figure(figsize=(14, 10), dpi=80)

    # colors_sampled = plt.cm.CMRmap(np.linspace(1, 0, 256))
    green_sampled = plt.cm.summer(np.linspace(1, 0, 256))
    red_sampled = plt.cm.autumn(np.linspace(0, 1, 256))

    max_score = abs(scores_absolute[-1])
    min_score = abs(scores_absolute[0])
    ratio_max = 255 / max_score
    ratio_min = 255 / min_score

    colors = [red_sampled[int(s * ratio_min)] if s < 0 else green_sampled[int(s * ratio_max)] for s in scores_absolute]

    plt.hlines(y=pseudos_absolute, xmin=0, xmax=scores_absolute, colors=colors, alpha=0.8, linewidth=5)

    if plot_text_values:
        tex_list = []
        for x, y, tex in zip(reversed(scores_absolute), reversed(pseudos_absolute), reversed(scores_absolute)):
            if tex not in tex_list:
                tex_list.append(tex)
                plt.text(x, y, " " + str(round(tex, 2)) + " ", horizontalalignment='right' if x < 0 else 'left',
                         verticalalignment='center', fontdict={'color': 'black', 'size': 8})

    plt.gca().set(ylabel='$Player$', xlabel='$Points$')
    plt.yticks(pseudos_absolute, pseudos_absolute, fontsize=11)
    plt.title('Bet score for players with at least ' + str(n) + " pronostics (" + str(number_of_players) + " players)",
              fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.4)
    plt.savefig("figures/bets_global_ranking_" + str(n) + ".png", bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 12))
    max_score = abs(scores_relative[-1])
    min_score = abs(scores_relative[0])
    ratio_max = 255 / max_score
    ratio_min = 255 / min_score

    plt.xlim(scores_relative[0] - 0.1, scores_relative[-1] + 0.1)

    colors = [red_sampled[int(s * ratio_min)] if s < 0 else green_sampled[int(s * ratio_max)] for s in scores_relative]

    plt.hlines(y=pseudos_relative, xmin=0, xmax=scores_relative, colors=colors, alpha=0.8, linewidth=5)

    if plot_text_values:
        tex_list = []
        for x, y, tex in zip(reversed(scores_relative), reversed(pseudos_relative), reversed(scores_relative)):
            if tex not in tex_list:
                tex_list.append(tex)
                plt.text(x, y, (" %+.1f %% " % float(100 * round(tex, 3))),
                         horizontalalignment='right' if x < 0 else 'left',
                         verticalalignment='center', fontdict={'color': 'black', 'size': 8})

    plt.gca().set(ylabel='$Player$', xlabel='Fortune')
    plt.yticks(pseudos_relative, pseudos_relative, fontsize=11)

    plt.draw()
    locs, _ = plt.xticks()
    x_labels = [(" %+d %% " % float(100 * round(s, 3))) for s in locs]
    plt.xticks(locs, x_labels, fontsize=11)
    plt.title('Relative bet score for players with at least ' + str(n) + " pronostics (" + str(
        number_of_players) + " players)", fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.4)
    plt.savefig("figures/relative_bets_global_ranking_" + str(n) + ".png", bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index

    pseudos = [x[0] for x in pseudos_score_and_number_of_pronostics]
    scores = [x[1] for x in pseudos_score_and_number_of_pronostics]
    number_of_pronostics = [x[2] for x in pseudos_score_and_number_of_pronostics]
    plt.scatter(number_of_pronostics, scores)

    texts = []
    for i, txt in enumerate(pseudos):
        texts.append(plt.text(number_of_pronostics[i] + 1, scores[i], txt, fontsize=9))

    # adjust_text(texts, only_move={'points':'y', 'texts':'y'})

    plt.gca().set(xlabel='$Nombre de pronostics$', ylabel='$Score$')
    plt.title('Score en fonction du nombre de pronostics', fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.4)

    alpha = np.array([[0.2, 1.],
                      [0.2, 1.]])

    gradient_image(ax, direction=0, extent=(0, 1, 0, 1), transform=ax.transAxes,
                   cmap=plt.cm.RdYlGn, cmap_range=(0.05, 0.95), alpha=alpha)
    plt.savefig("figures/bet_score_depending_on_pronostics_number.png", bbox_inches='tight')
    plt.show()


def gradient_image(ax, extent, direction=0.3, cmap_range=(0, 1), **kwargs):
    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])
    X = np.array([[v @ [1, 0], v @ [1, 1]],
                  [v @ [0, 0], v @ [0, 1]]])
    a, b = cmap_range
    X = a + (b - a) / X.max() * X

    im = ax.imshow(X, extent=extent, interpolation='gaussian',
                   vmin=0, vmax=1, **kwargs)
    return im


def get_bets_by_match(csv_path):
    pseudos_list = []
    odds = get_odds(csv_path)
    with open(csv_path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        number_of_days = len(next(csv_reader)) - 1
        for row in csv_reader:
            pseudo = row[0]
            if pseudo in csv_headers:
                continue
            pseudo = get_principal_pseudo(aliases, pseudo)

            if pseudo not in [x[0] for x in pseudos_list]:
                scores_by_day = np.zeros((number_of_days, 10))
                pseudos_list.append([pseudo, scores_by_day])

            pseudo_position = [index for index, row in enumerate(pseudos_list) if pseudo in row][0]

            for nb_day, (score, pronostic) in enumerate(zip(formatted_scores, row[1:])):
                for nb_match, (s, p) in enumerate(zip(score, eval(pronostic))):
                    if p == 0:
                        continue
                    elif s == p:
                        if p == "1":
                            pseudos_list[pseudo_position][1][nb_day][nb_match] = odds[nb_day][nb_match][0] - 1
                        if p == "N":
                            pseudos_list[pseudo_position][1][nb_day][nb_match] = odds[nb_day][nb_match][1] - 1
                        if p == "2":
                            pseudos_list[pseudo_position][1][nb_day][nb_match] = odds[nb_day][nb_match][2] - 1
                    else:
                        pseudos_list[pseudo_position][1][nb_day][nb_match] -= 1
    return pseudos_list


def get_bets_by_day(csv_path):
    bets_by_match = get_bets_by_match(csv_path)
    bets_by_day = []

    for bet_by_match in bets_by_match:
        pseudo = bet_by_match[0]
        bet = bet_by_match[1]
        bet_sum_on_day = []
        for b in bet:
            bet_sum_on_day.append(sum(b))
        bets_by_day.append([pseudo, bet_sum_on_day])
    return sorted(bets_by_day, key=lambda x: sum(x[1]), reverse=True)


def get_bets_global_ranking(csv_path):
    bets_by_day = get_bets_by_day(csv_path)
    return [[pseudo, sum(bet_score)] for pseudo, bet_score in bets_by_day]


def get_number_of_pronostics(csv_path):
    pseudos_list = []
    with open(csv_path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            pseudo = row[0]
            if pseudo in csv_headers:
                continue
            pseudo = get_principal_pseudo(aliases, pseudo)
            if pseudo not in [x[0] for x in pseudos_list]:
                pseudos_list.append([pseudo, 0])

            pseudo_position = [index for index, row in enumerate(pseudos_list) if pseudo in row][0]
            total_number_of_pronostics = 0
            for pronostic in row[1:]:
                for p in eval(pronostic):
                    if p != 0:
                        total_number_of_pronostics += 1
            pseudos_list[pseudo_position][1] += total_number_of_pronostics
    return sorted(pseudos_list, key=lambda x: x[1], reverse=True)


def get_pseudos_with_at_least_n_pronostics(csv_path, n=0):
    number_of_pronostics_list = get_number_of_pronostics(csv_path)
    pseudos_list = []
    for pseudo, number_of_pronostics in number_of_pronostics_list:
        if number_of_pronostics >= n:
            pseudos_list.append([pseudo, number_of_pronostics])
    return pseudos_list


def do_plot(ax):
    ax.plot([1, 2, 3], [4, 5, 6], 'k.')


def plot_khey_cotes(csv_path):
    with open(csv_path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row_header = row[0]
            if row_header == csv_headers[1]:
                break
        match_list = np.asarray([eval(x) for x in row[1:]])[..., :-1]

    odds = get_odds(csv_path)
    number_of_bets = get_number_of_bets(csv_path)

    days_number = len(number_of_bets)
    cols = int(math.sqrt(days_number))
    rows = int(math.ceil(days_number / cols))

    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(10 * cols, 10 * cols))
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    [s.set_visible(False) for s in plt.gca().spines.values()]

    background_color = "#054765"
    day_background_color = background_color
    fig.set_facecolor(background_color)
    plt.gca().set_facecolor(background_color)

    plt.title("Résultats et khey-cotes", fontdict={'size': 40, 'color': "white", 'fontweight': 'bold'}, y=1.03)

    for day in range(days_number):
        ax = fig.add_subplot(gs[day])
        bets_sum = [sum(n) for n in number_of_bets[day]]

        # fig, ax = plt.subplots(figsize=(9.2, 5))
        # ax.invert_yaxis()
        ax.set_facecolor(day_background_color)
        ax.xaxis.set_visible(False)

        bars_width = 1.15
        ax.set_xlim(0, bars_width)
        ax.set_title("J" + str(day + 1), fontdict={'size': 20, 'color': "white", 'fontweight': 'bold'})
        ax2 = ax.twinx()

        [s.set_visible(False) for s in ax.spines.values()]
        [s.set_visible(False) for s in ax2.spines.values()]

        widths_previous = [0.] * 10
        widths = widths_previous.copy()
        labels_names = ["1", "N", "2"]

        for y, x in enumerate(match_list[day][::-1]):
            team_home = str(x[0]).lower()
            team_away = str(x[1]).lower()
            xy = [0, y]
            fn = get_sample_data(os.path.join(res_dir, team_home + '.png'), asfileobj=False)
            arr_img = plt.imread(fn, format='png')
            imagebox = OffsetImage(arr_img, zoom=0.4)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, xy, frameon=False, xybox=(-0.035, xy[1]))
            ax.add_artist(ab)

            xy = [0, y]
            fn = get_sample_data(os.path.join(res_dir, team_away + '.png'), asfileobj=False)
            arr_img = plt.imread(fn, format='png')
            imagebox = OffsetImage(arr_img, zoom=0.4)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, xy, frameon=False, xybox=(1.185, xy[1]))
            ax.add_artist(ab)

        for i, colname in enumerate(labels_names):
            for j, (x, y) in enumerate(zip(number_of_bets[day][:, i], bets_sum)):
                widths[j] += x / y + 0.05

            # ax2.tick_params([str(m) for m in match_list[day]], colors='r')

            results = [str(m[2]) for m in match_list[day][::-1]]
            if i == 0:
                color = ["#8EE600" if x.strip() == "1" else "#CED4D9" for x in results]
            elif i == 1:
                color = ["#8EE600" if x.strip() == "N" else "#D4D9D0" for x in results]
            else:
                color = ["#8EE600" if x.strip() == "2" else "#CED4D9" for x in results]

            ax2.barh(["     " + str(m[1]) + "   " for m in match_list[day][::-1]], [0.] * 10, left=[0.] * 10,
                     height=0.7, color="red")
            ax.barh(["   " + str(m[0]) + "     " for m in match_list[day][::-1]], widths[::-1],
                    left=widths_previous[::-1], height=0.7, color=color, edgecolor = "grey")
            ax.tick_params(axis='y', colors='white', labelsize=16)
            ax2.tick_params(axis='y', colors='white', labelsize=16)
            xcenters = [(x + y) / 2 for x, y in zip(widths_previous, widths)]
            #
            # print(widths)
            number_of_bets_transpose_current_day = np.array(number_of_bets[day]).T.tolist()
            # print(number_of_bets[day])
            # print(number_of_bets_transpose_current_day)
            # print(widths)
            # print(number_of_bets[day][:, i])

            for y, (x, c) in enumerate(zip(xcenters[::-1], odds[day][:, i][::-1])):
                if not math.isnan(c):
                    ax.text(x, y, str(round(c, 2)), ha='center', va='center',
                            color="black", fontsize=10)
                else:
                    ax.text(x, y, "∅", ha='center', va='center',
                            color="black", fontsize=10)

            widths_previous = widths.copy()

    fig.tight_layout()
    plt.savefig("figures/khey_cotes.png", bbox_inches='tight', facecolor=background_color)
    plt.show()
    return


def plot_summary_by_day(csv_path, day):
    day -= 1
    with open(csv_path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row_header = row[0]
            if row_header == csv_headers[1]:
                break
        match_list = np.asarray([eval(x) for x in row[1:]])[..., :-1]

    odds = get_odds(csv_path)
    number_of_bets = get_number_of_bets(csv_path)

    # fig = plt.figure(figsize=(15, 15))

    division_size = 10

    fig, axs = plt.subplots(ncols=division_size, nrows=division_size, figsize=(15, 15))
    # fig, ax = plt.subplots(figsize=(15, 15))
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    [s.set_visible(False) for s in plt.gca().spines.values()]

    background_color = "#054765"
    day_background_color = background_color
    fig.set_facecolor(background_color)
    plt.gca().set_facecolor(background_color)

    bets_sum = [sum(n) for n in number_of_bets[day]]

    for ax_row in range(division_size):
        for ax in axs[ax_row, :]:
            ax.remove()

    gs = axs[0, 0].get_gridspec()

    ################################################## khey-cotes ##################################################
    ax = fig.add_subplot(gs[0:3, 3:7])
    ax.set_facecolor(day_background_color)
    ax.xaxis.set_visible(False)

    bars_width = 1.15
    ax.set_xlim(0, bars_width)
    ax.set_title("Bilan J" + str(day + 1), fontdict={'size': 20, 'color': "white", 'fontweight': 'bold'})
    ax2 = ax.twinx()

    [s.set_visible(False) for s in ax.spines.values()]
    [s.set_visible(False) for s in ax2.spines.values()]

    widths_previous = [0.] * 10
    widths = widths_previous.copy()
    labels_names = ["1", "N", "2"]

    for y, x in enumerate(match_list[day][::-1]):
        team_home = str(x[0]).lower()
        team_away = str(x[1]).lower()
        xy = [0, y]
        fn = get_sample_data(os.path.join(res_dir, team_home + '.png'), asfileobj=False)
        arr_img = plt.imread(fn, format='png')
        imagebox = OffsetImage(arr_img, zoom=0.4)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, xy, frameon=False, xybox=(-0.055, xy[1]))
        ax.add_artist(ab)

        xy = [0, y]
        fn = get_sample_data(os.path.join(res_dir, team_away + '.png'), asfileobj=False)
        arr_img = plt.imread(fn, format='png')
        imagebox = OffsetImage(arr_img, zoom=0.4)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, xy, frameon=False, xybox=(1.205, xy[1]))
        ax.add_artist(ab)

    for i, colname in enumerate(labels_names):
        for j, (x, y) in enumerate(zip(number_of_bets[day][:, i], bets_sum)):
            widths[j] += x / y + 0.05

        # ax2.tick_params([str(m) for m in match_list[day]], colors='r')

        results = [str(m[2]) for m in match_list[day][::-1]]
        if i == 0:
            color = ["#8EE600" if x.strip() == "1" else "#CED4D9" for x in results]
        elif i == 1:
            color = ["#8EE600" if x.strip() == "N" else "#D4D9D0" for x in results]
        else:
            color = ["#8EE600" if x.strip() == "2" else "#CED4D9" for x in results]

        ax2.barh(["     " + str(m[1]) + "   " for m in match_list[day][::-1]], [0.] * 10, left=[0.] * 10,
                 height=0.7, color="red")
        ax.barh(["   " + str(m[0]) + "     " for m in match_list[day][::-1]], widths[::-1],
                left=widths_previous[::-1], height=0.7, color=color, edgecolor = "grey")
        ax.tick_params(axis='y', colors='white', labelsize=16)
        ax2.tick_params(axis='y', colors='white', labelsize=16)

        colors_home = ["white"]*10
        colors_away = ["white"]*10

        for index, r in enumerate(results):
            if r.strip()=="1":
                colors_home[index] = "#04CD18"
                colors_away[index] = "#FA0000"
            elif r.strip()=="2":
                colors_home[index] = "#FA0000"
                colors_away[index] = "#04CD18"


        for xtick, color in zip(ax.get_yticklabels(), colors_home):
            xtick.set_color(color)
        for xtick, color in zip(ax2.get_yticklabels(), colors_away):
            xtick.set_color(color)

        xcenters = [(x + y) / 2 for x, y in zip(widths_previous, widths)]

        for y, (x, c) in enumerate(zip(xcenters[::-1], odds[day][:, i][::-1])):
            if not math.isnan(c):
                ax.text(x, y, str(round(c, 2)), ha='center', va='center',
                        color="black", fontsize=10)
            else:
                ax.text(x, y, "∅", ha='center', va='center',
                        color="black", fontsize=10)

        widths_previous = widths.copy()

    ############################################### classement journée ###############################################

    ax = fig.add_subplot(gs[3:, 0:5])
    ax.set_facecolor(background_color)
    ax.xaxis.set_visible(False)
    ax.set_title("Classement de la journée", fontdict={'size': 20, 'color': "white", 'fontweight': 'bold'})

    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    [s.set_visible(False) for s in plt.gca().spines.values()]

    height = 10
    colors_sampled = plt.cm.brg(np.linspace(0.85, 0.62, 11))
    colors_cell = ["#054D6E","#0E5678"]

    table_indexes = [[str(i)+"/10"] for i in range(height+1)]
    table = [[str(i)+"/10"] for i in range(height+1)]
    ranking_for_day = get_ranking_for_day(csv_path, day=day+1)


    for player_score in ranking_for_day:
        player= player_score[0]
        score= player_score[1]
        if score==0:
            continue
        table_indexes[score].append(player)

    for r,players_rank in enumerate(table_indexes):
        i=0
        rank_text=""
        while i<len(players_rank[1:]):
            i+=1
            if i%3==0:
                if i==len(players_rank[1:]):
                    rank_text +=players_rank[i]
                else:
                    rank_text +=players_rank[i]+"\n"
            else:
                rank_text +=players_rank[i]+"   "
        table[r].append(rank_text)

    table = [t for t in table if len(t[1])>0]

    height_ratio = 45


    day_table = ax.table(cellText=table[::-1], loc='center')
    day_table.scale(1.65, height_ratio/len(table))
    day_table.auto_set_font_size(False)

    cellDict=day_table.get_celld()
    for i in range(len(table)):
        cellDict[(i,0)].set_width(0.12)
        cellDict[(i,0)].set_color(colors_sampled[i])
        cellDict[(i,0)]._loc = 'center'
        cellDict[(i,0)].get_text().set_fontsize(20)
        # cellDict[(i,0)].auto_set_font_size()
        cellDict[(i,0)].get_text().set_color('white')
        cellDict[(i,0)].get_text().set_fontweight('bold')


    for i in range(len(table)):
        cellDict[(i,1)].set_color(colors_cell[int(i)/2==int(i/2)])
        # cellDict[(i,1)].set_width(0.14)
        cellDict[(i,1)]._loc = 'left'
        cellDict[(i,1)].get_text().set_fontsize(12)
        cellDict[(i,1)].PAD=0.05

        cellDict[(i,1)].get_text().set_color('white')
        cellDict[(i,1)].get_text().set_fontweight('bold')


    ############################################### classement général ###############################################

    ax = fig.add_subplot(gs[3:, 5:])
    ax.set_facecolor(background_color)
    ax.xaxis.set_visible(False)
    ax.set_title("Classement général", fontdict={'size': 20, 'color': "white", 'fontweight': 'bold'})

    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    [s.set_visible(False) for s in plt.gca().spines.values()]

    ranking_general = get_global_ranking(csv_path,day+1)
    max_points = ranking_general[0][1]
    min_points = max(1,int(0.22*max_points-1))
    table_indexes = [[i] for i in range(min_points,max_points+1)]
    table = [[i] for i in range(min_points,max_points+1)]

    colors_sampled = plt.cm.brg(np.linspace(0.85, 0.62, max_points+1))
    colors_cell = ["#054D6E","#0E5678"]

    for player_score in ranking_general:
        player= player_score[0]
        score= player_score[1]
        if score<min_points:
            continue
        table_indexes[score-min_points].append(player)

    for r,players_rank in enumerate(table_indexes):
        i=0
        rank_text=""
        while i<len(players_rank[1:]):
            i+=1
            if i%3==0:
                if i==len(players_rank[1:]):
                    rank_text +=players_rank[i]
                else:
                    rank_text +=players_rank[i]+"\n"
            else:
                rank_text +=players_rank[i]+"   "
        table[r].append(rank_text)



    day_table = ax.table(cellText=table[::-1], loc='center')
    day_table.scale(1.65, height_ratio/len(table))
    day_table.auto_set_font_size(False)

    cellDict=day_table.get_celld()
    for i in range(len(table)):
        cellDict[(i,0)].set_width(0.07)
        cellDict[(i,0)].set_color(colors_sampled[i])
        cellDict[(i,0)]._loc = 'center'
        cellDict[(i,0)].get_text().set_fontsize(12)
        # cellDict[(i,0)].auto_set_font_size()
        cellDict[(i,0)].get_text().set_color('white')
        cellDict[(i,0)].get_text().set_fontweight('bold')


    for i in range(len(table)):
        cellDict[(i,1)].set_color(colors_cell[int(i)/2==int(i/2)])
        cellDict[(i,1)]._loc = 'left'
        cellDict[(i,1)].get_text().set_fontsize(10)
        cellDict[(i,1)].PAD=0.05
        cellDict[(i,1)].get_text().set_color('white')
        cellDict[(i,1)].get_text().set_fontweight('bold')

####################################################  fin  #######################################################

    fig.tight_layout()
    plt.savefig("figures/bilans/bilan_J" + str(day + 1) + ".png", bbox_inches='tight', facecolor=background_color)
    plt.show()
    return

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    data = np.asarray(data)
    if not ax:
        ax = plt.gca()
    im = ax.imshow(data,  aspect='auto', **kwargs)

    cbar = ax.figure.colorbar(im, ax=ax, aspect = 40,**cbar_kw)
    cbar.set_ticks([t for t in cbar.ax.get_yticks()])
    cbar.set_ticklabels([str(int(t*100))+"%" for t in cbar.ax.get_yticks()])
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontdict={'color': 'black', 'size': 16})
    cbar.ax.tick_params(labelsize=12)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
             rotation_mode="anchor")

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="#000924", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis='x', labelsize = 11)
    ax.tick_params(axis='y', labelsize = 14)

    return im, cbar

def heatmap_total(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    data = np.asarray(data)
    if not ax:
        ax = plt.gca()
    im = ax.imshow(data,  aspect='auto', **kwargs)

    cbar = ax.figure.colorbar(im, ax=ax, aspect = 100)
    cbar.set_ticks([])

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="#000924", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis='x', labelsize = 11)
    ax.tick_params(axis='y', labelsize = 14)

    return im

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_matrix_team_player(csv_path):
    pseudos_list = []
    match_teams = []

    with open(csv_path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            pseudo = row[0]
            if pseudo == csv_headers[1]:
                match_teams = [eval(r) for r in row[1:]]
                for i,match_days in enumerate(match_teams):
                    for j,match in enumerate(match_days):
                        match_teams[i][j]= match[:-2]

            if pseudo in csv_headers:
                continue
            pseudo = get_principal_pseudo(aliases, pseudo)
            if pseudo not in [x[0] for x in pseudos_list]:
                pseudos_list.append([pseudo, np.zeros((20,2)).tolist() ])

            pseudo_position = [index for index, row in enumerate(pseudos_list) if pseudo in row][0]

            for match, pronostic in zip(match_teams, row[1:]):
                for m,p in zip(match,eval(pronostic)):
                    if str(p).strip()=='1':
                        team_bet_on = m[0]
                        team_bet_against = m[1]
                        index_team_bet_on = [i for i, t in enumerate(teams_dict) if t==team_bet_on][0]
                        index_team_bet_against = [i for i, t in enumerate(teams_dict) if t==team_bet_against][0]
                        pseudos_list[pseudo_position][1][index_team_bet_on][0] +=1
                        pseudos_list[pseudo_position][1][index_team_bet_against][1] +=1
                    elif str(p).strip()=='N':
                        team_1 = m[0]
                        team_2 = m[1]
                        index_team_1 = [i for i, t in enumerate(teams_dict) if t==team_1][0]
                        index_team_2 = [i for i, t in enumerate(teams_dict) if t==team_2][0]
                        pseudos_list[pseudo_position][1][index_team_1][0] +=0.5
                        pseudos_list[pseudo_position][1][index_team_1][1] +=0.5
                        pseudos_list[pseudo_position][1][index_team_2][0] +=0.5
                        pseudos_list[pseudo_position][1][index_team_2][1] +=0.5
                    elif str(p).strip()=='2':
                        team_bet_on = m[1]
                        team_bet_against = m[0]
                        index_team_bet_on = [i for i, t in enumerate(teams_dict) if t==team_bet_on][0]
                        index_team_bet_against = [i for i, t in enumerate(teams_dict) if t==team_bet_against][0]
                        pseudos_list[pseudo_position][1][index_team_bet_on][0] +=1
                        pseudos_list[pseudo_position][1][index_team_bet_against][1] +=1

    indexes_to_pop =[]
    for i, pseudo_match in enumerate(pseudos_list):
        for j, match in enumerate(pseudo_match[1]):
            if (match[0]+match[1] <1):
                indexes_to_pop.append(i)
                break


    pseudos_list = [i for j, i in enumerate(pseudos_list) if j not in indexes_to_pop]
    pseudos_list = sorted(pseudos_list, key=lambda x: sum(sum(x[1],[])), reverse=True)

    pseudos_list_scaled=[]
    for i, pseudo_match in enumerate(pseudos_list):
        pseudos_list_scaled.append([pseudos_list[i][0],(np.zeros((20,1))).tolist()])
        for j, match in enumerate(pseudo_match[1]):
            pseudos_list_scaled[i][1][j]= round(match[0]/(match[0]+match[1]),2)

    data = [x[1] for x in pseudos_list_scaled]
    kheys = [x[0] for x in pseudos_list_scaled]
    teams = [t for t in teams_dict]

    grid = dict(height_ratios=[len(pseudos_list_scaled), 1])
    fig, axs = plt.subplots(ncols=1, nrows=2,figsize=(8, 12), gridspec_kw = grid)

    axs[0].set_title("Confiance khey/équipe", fontdict={'size': 20,'fontweight': 'bold'}, y=1.11)


    min_data = np.min(data)
    max_data = np.max(data)

    cmap = plt.get_cmap('RdYlGn')
    colors = cmap(np.linspace(min_data, max_data, cmap.N))
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list('my_color_map', colors)

    im, cbar = heatmap(data, kheys, teams, ax=axs[0],
                           cmap=color_map, cbarlabel="Confiance")

    pseudo_bets = [p[1] for p in pseudos_list]
    mean_list=np.zeros((20,2)).tolist()
    for t in pseudo_bets:
        mean_list = [[mean_list[i][j] + t[i][j]  for j in range
        (len(mean_list[0]))] for i in range(len(mean_list))]

    mean_list = [x[0]/(x[0]+x[1]) for x in mean_list]

    max_mean = max(mean_list)
    min_mean = min(mean_list)

    cmap = plt.get_cmap('RdYlGn')
    colors_mean = cmap(np.linspace(min_mean, max_mean, cmap.N))
    color_map_mean = matplotlib.colors.LinearSegmentedColormap.from_list('my_color_map', colors_mean)

    im= heatmap_total([mean_list], ["Moyenne"], teams, ax=axs[1],
                       cmap=color_map_mean)

    fig.tight_layout()
    plt.savefig("figures/confiance_khey_equipe.png", bbox_inches='tight')
    plt.show()


def plot_matrix_team_player_relative(csv_path):
    pseudos_list = []
    match_teams = []

    with open(csv_path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            pseudo = row[0]
            if pseudo == csv_headers[1]:
                match_teams = [eval(r) for r in row[1:]]
                for i,match_days in enumerate(match_teams):
                    for j,match in enumerate(match_days):
                        match_teams[i][j]= match[:-2]

            if pseudo in csv_headers:
                continue
            pseudo = get_principal_pseudo(aliases, pseudo)
            if pseudo not in [x[0] for x in pseudos_list]:
                pseudos_list.append([pseudo, np.zeros((20,2)).tolist() ])

            pseudo_position = [index for index, row in enumerate(pseudos_list) if pseudo in row][0]

            for match, pronostic in zip(match_teams, row[1:]):
                for m,p in zip(match,eval(pronostic)):
                    if str(p).strip()=='1':
                        team_bet_on = m[0]
                        team_bet_against = m[1]
                        index_team_bet_on = [i for i, t in enumerate(teams_dict) if t==team_bet_on][0]
                        index_team_bet_against = [i for i, t in enumerate(teams_dict) if t==team_bet_against][0]
                        pseudos_list[pseudo_position][1][index_team_bet_on][0] +=1
                        pseudos_list[pseudo_position][1][index_team_bet_against][1] +=1
                    elif str(p).strip()=='N':
                        team_1 = m[0]
                        team_2 = m[1]
                        index_team_1 = [i for i, t in enumerate(teams_dict) if t==team_1][0]
                        index_team_2 = [i for i, t in enumerate(teams_dict) if t==team_2][0]
                        pseudos_list[pseudo_position][1][index_team_1][0] +=0.5
                        pseudos_list[pseudo_position][1][index_team_1][1] +=0.5
                        pseudos_list[pseudo_position][1][index_team_2][0] +=0.5
                        pseudos_list[pseudo_position][1][index_team_2][1] +=0.5
                    elif str(p).strip()=='2':
                        team_bet_on = m[1]
                        team_bet_against = m[0]
                        index_team_bet_on = [i for i, t in enumerate(teams_dict) if t==team_bet_on][0]
                        index_team_bet_against = [i for i, t in enumerate(teams_dict) if t==team_bet_against][0]
                        pseudos_list[pseudo_position][1][index_team_bet_on][0] +=1
                        pseudos_list[pseudo_position][1][index_team_bet_against][1] +=1

    indexes_to_pop =[]
    for i, pseudo_match in enumerate(pseudos_list):
        for j, match in enumerate(pseudo_match[1]):
            if (match[0]+match[1] <1):
                indexes_to_pop.append(i)
                break

    pseudos_list = [i for j, i in enumerate(pseudos_list) if j not in indexes_to_pop]
    pseudos_list = sorted(pseudos_list, key=lambda x: sum(sum(x[1],[])), reverse=True)

    pseudos_list_scaled=[]
    for i, pseudo_match in enumerate(pseudos_list):
        pseudos_list_scaled.append([pseudos_list[i][0],(np.zeros((20,1))).tolist()])
        for j, match in enumerate(pseudo_match[1]):
            pseudos_list_scaled[i][1][j]= round(match[0]/(match[0]+match[1]),2)


    data = [x[1] for x in pseudos_list_scaled]
    kheys = [x[0] for x in pseudos_list_scaled]
    teams = [t for t in teams_dict]

    grid = dict(height_ratios=[len(pseudos_list_scaled), 1])
    fig, axs = plt.subplots(ncols=1, nrows=2,figsize=(8, 12), gridspec_kw = grid)

    axs[0].set_title("Confiance khey/équipe (écart à la moyenne)", fontdict={'size': 20,'fontweight': 'bold'}, y=1.11)


    pseudo_bets = [p[1] for p in pseudos_list]
    mean_list=np.zeros((20,2)).tolist()
    for t in pseudo_bets:
        mean_list = [[mean_list[i][j] + t[i][j]  for j in range
        (len(mean_list[0]))] for i in range(len(mean_list))]

    mean_list = [x[0]/(x[0]+x[1]) for x in mean_list]

    for i, data_player in enumerate(data):
        data[i] = [d-m for d,m in zip(data_player, mean_list)]

    im, cbar = heatmap(data, kheys, teams, ax=axs[0],
                       cmap="RdYlGn", cbarlabel="Confiance")

    max_mean = max(mean_list)
    min_mean = min(mean_list)

    cmap = plt.get_cmap('RdYlGn')
    colors_mean = cmap(np.linspace(min_mean, max_mean, cmap.N))
    color_map_mean = matplotlib.colors.LinearSegmentedColormap.from_list('my_color_map', colors_mean)


    im= heatmap_total([mean_list], ["Moyenne"], teams, ax=axs[1],
                      cmap=color_map_mean)

    fig.tight_layout()
    plt.savefig("figures/confiance_khey_equipe_relatif.png", bbox_inches='tight')
    plt.show()


def plot_matrix_team_player_success(csv_path):
    pseudos_list = []
    match_teams = []

    with open(csv_path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            pseudo = row[0]
            if pseudo == csv_headers[1]:
                match_teams = [eval(r) for r in row[1:]]
                for i,match_days in enumerate(match_teams):
                    for j,match in enumerate(match_days):
                        match_teams[i][j]= match[:-2]

            if pseudo in csv_headers:
                continue
            pseudo = get_principal_pseudo(aliases, pseudo)
            if pseudo not in [x[0] for x in pseudos_list]:
                pseudos_list.append([pseudo, np.zeros((20,2)).tolist() ])

            pseudo_position = [index for index, row in enumerate(pseudos_list) if pseudo in row][0]


            for match, pronostic, score in zip(match_teams, row[1:], formatted_scores):
                for m,p,s in zip(match,eval(pronostic), score):
                    if not p in scores_list:
                        continue
                    team_home = m[0]
                    team_away = m[1]
                    index_team_home = [i for i, t in enumerate(teams_dict) if t==team_home][0]
                    index_team_away = [i for i, t in enumerate(teams_dict) if t==team_away][0]

                    if str(p).strip()==str(s).strip():
                        pseudos_list[pseudo_position][1][index_team_home][0] +=1
                        pseudos_list[pseudo_position][1][index_team_away][0] +=1
                    else:
                        pseudos_list[pseudo_position][1][index_team_home][1] +=1
                        pseudos_list[pseudo_position][1][index_team_away][1] +=1

    indexes_to_pop =[]
    for i, pseudo_match in enumerate(pseudos_list):
        for j, match in enumerate(pseudo_match[1]):
            if (match[0]+match[1] <1):
                indexes_to_pop.append(i)
                break

    pseudos_list = [i for j, i in enumerate(pseudos_list) if j not in indexes_to_pop]
    pseudos_list = sorted(pseudos_list, key=lambda x: sum(sum(x[1],[])), reverse=True)

    pseudos_list_scaled=[]
    for i, pseudo_match in enumerate(pseudos_list):
        pseudos_list_scaled.append([pseudos_list[i][0],(np.zeros((20,1))).tolist()])
        for j, match in enumerate(pseudo_match[1]):
            pseudos_list_scaled[i][1][j]= round(match[0]/(match[0]+match[1]),2)


    data = [x[1] for x in pseudos_list_scaled]
    kheys = [x[0] for x in pseudos_list_scaled]
    teams = [t for t in teams_dict]

    grid = dict(height_ratios=[len(pseudos_list_scaled), 1])
    fig, axs = plt.subplots(ncols=1, nrows=2,figsize=(8, 12), gridspec_kw = grid)

    axs[0].set_title("Réussite khey/équipe", fontdict={'size': 20,'fontweight': 'bold'}, y=1.11)

    min_data = np.min(data)
    max_data = np.max(data)

    cmap = plt.get_cmap('RdYlGn')
    colors = cmap(np.linspace(min_data, max_data, cmap.N))
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list('my_color_map', colors)

    im, cbar = heatmap(data, kheys, teams, ax=axs[0],
                       cmap=color_map, cbarlabel="Taux de pronostics corrects")

    pseudo_bets = [p[1] for p in pseudos_list]
    mean_list=np.zeros((20,2)).tolist()
    for t in pseudo_bets:
        mean_list = [[mean_list[i][j] + t[i][j]  for j in range
        (len(mean_list[0]))] for i in range(len(mean_list))]

    mean_list = [x[0]/(x[0]+x[1]) for x in mean_list]

    max_mean = max(mean_list)
    min_mean = min(mean_list)

    cmap = plt.get_cmap('RdYlGn')
    colors_mean = cmap(np.linspace(min_mean, max_mean, cmap.N))
    color_map_mean = matplotlib.colors.LinearSegmentedColormap.from_list('my_color_map', colors_mean)


    im= heatmap_total([mean_list], ["Moyenne"], teams, ax=axs[1],
                          cmap=color_map_mean)

    fig.tight_layout()
    plt.savefig("figures/reussite_khey_equipe.png", bbox_inches='tight')
    plt.show()


def plot_matrix_team_player_success_relative(csv_path):
    pseudos_list = []
    match_teams = []

    with open(csv_path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            pseudo = row[0]
            if pseudo == csv_headers[1]:
                match_teams = [eval(r) for r in row[1:]]
                for i,match_days in enumerate(match_teams):
                    for j,match in enumerate(match_days):
                        match_teams[i][j]= match[:-2]

            if pseudo in csv_headers:
                continue
            pseudo = get_principal_pseudo(aliases, pseudo)
            if pseudo not in [x[0] for x in pseudos_list]:
                pseudos_list.append([pseudo, np.zeros((20,2)).tolist() ])

            pseudo_position = [index for index, row in enumerate(pseudos_list) if pseudo in row][0]


            for match, pronostic, score in zip(match_teams, row[1:], formatted_scores):
                for m,p,s in zip(match,eval(pronostic), score):
                    if not p in scores_list:
                        continue
                    team_home = m[0]
                    team_away = m[1]
                    index_team_home = [i for i, t in enumerate(teams_dict) if t==team_home][0]
                    index_team_away = [i for i, t in enumerate(teams_dict) if t==team_away][0]

                    if str(p).strip()==str(s).strip():
                        pseudos_list[pseudo_position][1][index_team_home][0] +=1
                        pseudos_list[pseudo_position][1][index_team_away][0] +=1
                    else:
                        pseudos_list[pseudo_position][1][index_team_home][1] +=1
                        pseudos_list[pseudo_position][1][index_team_away][1] +=1

    indexes_to_pop =[]
    for i, pseudo_match in enumerate(pseudos_list):
        for j, match in enumerate(pseudo_match[1]):
            if (match[0]+match[1] <1):
                indexes_to_pop.append(i)
                break

    pseudos_list = [i for j, i in enumerate(pseudos_list) if j not in indexes_to_pop]
    pseudos_list = sorted(pseudos_list, key=lambda x: sum(sum(x[1],[])), reverse=True)

    pseudos_list_scaled=[]
    for i, pseudo_match in enumerate(pseudos_list):
        pseudos_list_scaled.append([pseudos_list[i][0],(np.zeros((20,1))).tolist()])
        for j, match in enumerate(pseudo_match[1]):
            pseudos_list_scaled[i][1][j]= round(match[0]/(match[0]+match[1]),2)


    data = [x[1] for x in pseudos_list_scaled]
    kheys = [x[0] for x in pseudos_list_scaled]
    teams = [t for t in teams_dict]

    grid = dict(height_ratios=[len(pseudos_list_scaled), 1])
    fig, axs = plt.subplots(ncols=1, nrows=2,figsize=(8, 12), gridspec_kw = grid)

    axs[0].set_title("Réussite khey/équipe (écart à la moyenne)", fontdict={'size': 20,'fontweight': 'bold'}, y=1.11)

    min_data = np.min(data)
    max_data = np.max(data)

    cmap = plt.get_cmap('RdYlGn')
    colors = cmap(np.linspace(min_data, max_data, cmap.N))
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list('my_color_map', colors)


    pseudo_bets = [p[1] for p in pseudos_list]
    mean_list=np.zeros((20,2)).tolist()
    for t in pseudo_bets:
        mean_list = [[mean_list[i][j] + t[i][j]  for j in range
        (len(mean_list[0]))] for i in range(len(mean_list))]

    mean_list = [x[0]/(x[0]+x[1]) for x in mean_list]

    for i, data_player in enumerate(data):
        data[i] = [d-m for d,m in zip(data_player, mean_list)]

    im, cbar = heatmap(data, kheys, teams, ax=axs[0],
                       cmap=color_map, cbarlabel="Taux de pronostics corrects, écart à la moyenne")

    pseudo_bets = [p[1] for p in pseudos_list]
    mean_list=np.zeros((20,2)).tolist()
    for t in pseudo_bets:
        mean_list = [[mean_list[i][j] + t[i][j]  for j in range
        (len(mean_list[0]))] for i in range(len(mean_list))]

    mean_list = [x[0]/(x[0]+x[1]) for x in mean_list]

    max_mean = max(mean_list)
    min_mean = min(mean_list)

    cmap = plt.get_cmap('RdYlGn')
    colors_mean = cmap(np.linspace(min_mean, max_mean, cmap.N))
    color_map_mean = matplotlib.colors.LinearSegmentedColormap.from_list('my_color_map', colors_mean)


    im= heatmap_total([mean_list], ["Moyenne"], teams, ax=axs[1],
                      cmap=color_map_mean)

    fig.tight_layout()
    plt.savefig("figures/reussite_khey_equipe_relatif.png", bbox_inches='tight')
    plt.show()


def plot_matrix_team_player_coting(csv_path):
    pseudos_list = []
    match_teams = []

    with open(csv_path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            pseudo = row[0]
            if pseudo == csv_headers[1]:
                match_teams = [eval(r) for r in row[1:]]
                for i,match_days in enumerate(match_teams):
                    for j,match in enumerate(match_days):
                        match_teams[i][j]= match[:-2]

            if pseudo in csv_headers:
                continue
            pseudo = get_principal_pseudo(aliases, pseudo)
            if pseudo not in [x[0] for x in pseudos_list]:
                pseudos_list.append([pseudo, np.zeros((20,2)).tolist() ])

            pseudo_position = [index for index, row in enumerate(pseudos_list) if pseudo in row][0]


            for match, pronostic, score in zip(match_teams, row[1:], formatted_scores):
                for m,p,s in zip(match,eval(pronostic), score):
                    if not p in scores_list:
                        continue
                    team_home = m[0]
                    team_away = m[1]
                    index_team_home = [i for i, t in enumerate(teams_dict) if t==team_home][0]
                    index_team_away = [i for i, t in enumerate(teams_dict) if t==team_away][0]

                    pseudos_list[pseudo_position][1][index_team_home][0] +=1
                    pseudos_list[pseudo_position][1][index_team_away][0] +=1

                    if str(p).strip() == "1" and str(s).strip() != "1":
                        pseudos_list[pseudo_position][1][index_team_home][1] +=1
                        pseudos_list[pseudo_position][1][index_team_away][1] -=1

                    if str(p).strip() == "2" and str(s).strip() != "2":
                        pseudos_list[pseudo_position][1][index_team_home][1] -=1
                        pseudos_list[pseudo_position][1][index_team_away][1] +=1

                    if str(p).strip() == "N":
                        if str(s).strip() != "1":
                            pseudos_list[pseudo_position][1][index_team_home][1] -=1
                            pseudos_list[pseudo_position][1][index_team_away][1] +=1
                        elif str(s).strip() != "2":
                            pseudos_list[pseudo_position][1][index_team_home][1] +=1
                            pseudos_list[pseudo_position][1][index_team_away][1] -=1

    indexes_to_pop =[]
    for i, pseudo_match in enumerate(pseudos_list):
        for j, match in enumerate(pseudo_match[1]):
            if (match[0] <1):
                indexes_to_pop.append(i)
                break

    pseudos_list = [i for j, i in enumerate(pseudos_list) if j not in indexes_to_pop]
    pseudos_list = sorted(pseudos_list, key=lambda x: sum(sum(x[1],[])), reverse=True)

    pseudos_list_scaled=[]
    for i, pseudo_match in enumerate(pseudos_list):
        pseudos_list_scaled.append([pseudos_list[i][0],(np.zeros((20,1))).tolist()])
        for j, match in enumerate(pseudo_match[1]):
            pseudos_list_scaled[i][1][j]= round(match[1]/(match[0]),2)


    data = [x[1] for x in pseudos_list_scaled]
    kheys = [x[0] for x in pseudos_list_scaled]
    teams = [t for t in teams_dict]

    grid = dict(height_ratios=[len(pseudos_list_scaled), 1])
    fig, axs = plt.subplots(ncols=1, nrows=2,figsize=(8, 12), gridspec_kw = grid)

    axs[0].set_title("Equipes surcotées/sous-cotées", fontdict={'size': 20,'fontweight': 'bold'}, y=1.11)

    min_data = np.min(data)
    max_data = np.max(data)

    cmap = plt.get_cmap('bwr')
    colors = cmap(np.linspace(min_data/2+0.5, max_data/2+0.5, cmap.N))
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list('my_color_map', colors)

    im, cbar = heatmap(data, kheys, teams, ax=axs[0],
                       cmap=color_map, cbarlabel="Taux de surcotage/sous-cotage")

    pseudo_bets = [p[1] for p in pseudos_list]
    mean_list=np.zeros((20,2)).tolist()
    for t in pseudo_bets:
        mean_list = [[mean_list[i][j] + t[i][j]  for j in range
        (len(mean_list[0]))] for i in range(len(mean_list))]

    mean_list = [x[1]/x[0] for x in mean_list]
    mean_list = [(x+1)/2 for x in mean_list]

    max_mean = max(mean_list)
    min_mean = min(mean_list)

    cmap = plt.get_cmap('bwr')
    colors_mean = cmap(np.linspace(min_mean, max_mean, cmap.N))
    color_map_mean = matplotlib.colors.LinearSegmentedColormap.from_list('my_color_map', colors_mean)


    im= heatmap_total([mean_list], ["Moyenne"], teams, ax=axs[1],
                      cmap=color_map_mean)

    fig.tight_layout()
    plt.savefig("figures/surcote_souscote.png", bbox_inches='tight')
    plt.show()

def write_results(day_list, global_list, day):
    with open(csv_path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row_header = row[0]
            if row_header == csv_headers[1]:
                break
    csv_file.close()

    match_list = np.asarray([eval(x) for x in row[1:]])[..., :-1]

    file = open("logs/resultats_J"+str(day)+".txt","w", encoding="utf-8")
    file.write( "'''Résultat des matchs''' \n\n")

    for line in match_list[day-1]:
        team_home= teams_dict[line[0]][0]
        team_away=  teams_dict[line[1]][0]
        result= line[2]
        file.write(str(team_home)+" / "+str(team_away)+" = "+str(result)+ "\n")

    file.write( "\n'''Classement de la journée "+str(day)+" :''' \n")

    previous_score=11
    for line in day_list:
        pseudo = line[0]
        score = line[1]
        if score==0:
            break
        if previous_score != score:
            file.write( "\n'''"+str(score)+"/10'''\n")

        file.write(pseudo+"\n")
        previous_score = score


    file.write( "\n'''Classement général après la journée "+str(day)+" :''' \n")

    previous_score=99999
    for line in global_list:
        pseudo = line[0]
        score = line[1]
        if score==0:
            break
        if previous_score != score:
            file.write( "\n'''"+str(score)+"'''\n")

        file.write(pseudo+"\n")
        previous_score = score





global_ranking = get_global_ranking(csv_path)
total_number_of_players = len(global_ranking)

print("Nombre total de joueurs :", total_number_of_players)

print("\nKheys les plus assidus")
print(get_number_of_pronostics(csv_path))
print("\nClassement général:")
print(global_ranking)
print("\nScore par journée:")
print(get_score_by_day(csv_path))
print("\nScore cumulé par journée:")
print(get_cumulative_score(csv_path))

day = 1
print("\nClassement de la journée " + str(day) + " :")
print(get_ranking_for_day(csv_path, day=day))



odds = get_odds(csv_path)
number_of_bets = get_number_of_bets(csv_path)

# get_bets_by_day(csv_path)
# print(get_bets_by_match(csv_path))
# print(get_bets_by_day(csv_path))
# print(get_bets_global_ranking(csv_path))

"""
"""
plot_top_players_evolution(csv_path, number_of_players=15)
plot_ranking(csv_path, number_of_players=50, plot_text_values=True)
# if (number_of_d)
plot_ranking_stacked(csv_path, number_of_players=50)
plot_bets_global_ranking(csv_path, n=1, plot_text_values=True)
plot_score_depending_on_pronostic_number(csv_path, n=1)
for d in range(day,day+1):
    write_results(get_ranking_for_day(csv_path, day=d), get_global_ranking(csv_path, day=d), d)
    plot_summary_by_day(csv_path, day=d)
plot_khey_cotes(csv_path)
plot_matrix_team_player(csv_path)
plot_matrix_team_player_relative(csv_path)
plot_matrix_team_player_success(csv_path)
plot_matrix_team_player_success_relative(csv_path)
plot_matrix_team_player_coting(csv_path)


""""
A CHANGER :
    - PARSING DU <p>
    - +2 -2 pour les surcotages/sous-cotages

"""