import matplotlib.pyplot as plt
import os
import pandas as pd
import plotly.graph_objects as go

from collections import Counter
from palettable.colorbrewer.qualitative import Pastel1_7
from utils import IMG_DIR
from wordcloud import WordCloud

def analysis(pred):
    
    # box plots
    positive = pred['predictions'].value_counts()[1]
    negative = pred['predictions'].value_counts()[0]
    ratio_positive = int(positive/(positive+negative)*100)
    ratio_negative = int(100 - ratio_positive)
    top_labels = ['Positive', 'Negative']

    colors = ['orange', 'black']

    x_data = [[ratio_positive, ratio_negative]]

    y_data = ['Sentiments']

    fig = go.Figure()

    for i in range(0, len(x_data[0])):
        for xd, yd in zip(x_data, y_data):
            fig.add_trace(go.Bar(
                x=[xd[i]], y=[yd],
                orientation='h',
                marker=dict(
                    color=colors[i],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
            ))

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            domain=[0.15, 1]
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        barmode='stack',
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        margin=dict(l=120, r=10, t=140, b=80),
        showlegend=False,
        title_text=f"Ratio Between Positive and Negative Sentiments ({int(positive)+int(negative)} reviews)"
    )

    annotations = []

    for yd, xd in zip(y_data, x_data):
        # labeling the y-axis
        annotations.append(dict(xref='paper', yref='y',
                                x=0.14, y=yd,
                                xanchor='right',
                                text=str(yd),
                                font=dict(family='Arial', size=14,
                                        color='rgb(67, 67, 67)'),
                                showarrow=False, align='right'))
        # labeling the first percentage of each bar (x_axis)
        annotations.append(dict(xref='x', yref='y',
                                x=xd[0] / 2, y=yd,
                                text=str(xd[0]) + '%',
                                font=dict(family='Arial', size=20,
                                        color='yellow'),
                                showarrow=False))
        # labeling the first Likert scale (on the top)
        if yd == y_data[-1]:
            annotations.append(dict(xref='x', yref='paper',
                                    x=xd[0] / 2, y=1.1,
                                    text=top_labels[0],
                                    font=dict(family='Arial', size=14,
                                            color='black'),
                                    showarrow=False))
        space = xd[0]
        for i in range(1, len(xd)):
                # labeling the rest of percentages for each bar (x_axis)
                annotations.append(dict(xref='x', yref='y',
                                        x=space + (xd[i]/2), y=yd,
                                        text=str(xd[i]) + '%',
                                        font=dict(family='Arial', size=20,
                                                color='yellow'),
                                        showarrow=False))
                # labeling the Likert scale
                if yd == y_data[-1]:
                    annotations.append(dict(xref='x', yref='paper',
                                            x=space + (xd[i]/2), y=1.1,
                                            text=top_labels[i],
                                            font=dict(family='Arial', size=14,
                                                    color='black'),
                                            showarrow=False))
                space += xd[i]

    fig.update_layout(annotations=annotations)

    filename = "ratio_latest.jpeg"
    fig.write_image(os.path.join(IMG_DIR, filename))
    #fig.show()

    # Word Clouds for each sentiment
    positive_reviews = pred[pred['predictions']==1]['text']
    negative_reviews = pred[pred['predictions']==0]['text']

    pos_text = " ".join(i for i in positive_reviews)
    wordcloud = WordCloud(width = 800,
                      height = 800,
                      background_color="white").generate(pos_text)
    filename_pos_rev = "wc_pos_rev_latest.png"
    wordcloud.to_file((os.path.join(IMG_DIR, filename_pos_rev)))
    
    neg_text = " ".join(i for i in negative_reviews)
    wordcloud = WordCloud(width = 800,
                      height = 800,
                      background_color="white").generate(neg_text)
    filename_neg_rev = "wc_neg_rev_latest.png"
    wordcloud.to_file((os.path.join(IMG_DIR, filename_neg_rev)))

    # Text circular graph for both sentiements
    positive_reviews_text = text = " ".join(i for i in positive_reviews)
    negative_reviews_text = text = " ".join(i for i in negative_reviews)

    # top 20 most common positive words
    positive_word_counts = positive_reviews_text.split(" ")
    pos_counter = Counter(positive_word_counts)
    
    pos_counts = []
    pos_words = []
    for i in range(len(pos_counter)):
        if i == 20:
            break
        pos_words.append(pos_counter.most_common()[i][0])
        pos_counts.append(pos_counter.most_common()[i][1])

    
    # top 20 most common negative words
    negative_word_counts = negative_reviews_text.split(" ")
    neg_counter = Counter(negative_word_counts)

    neg_counts = []
    neg_words = []
    for i in range(len(neg_counter)):
        if i == 20:
            break
        neg_words.append(neg_counter.most_common()[i][0])
        neg_counts.append(neg_counter.most_common()[i][1])



    positive_word_counts = pd.DataFrame({"words": pos_words, "counts": pos_counts})
    negative_word_counts = pd.DataFrame({"words": neg_words, "counts": neg_counts})

    plt.figure(figsize=(5,5))
    my_circle=plt.Circle((0,0), 0.6, color='white')
    plt.pie(positive_word_counts['counts'], labels=positive_word_counts['words'], colors=Pastel1_7.hex_colors)
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    filename_top_20_pos = "top_20_pos_latest.png"
    plt.savefig(os.path.join(IMG_DIR, filename_top_20_pos), dpi = 300)
    plt.close()

    plt.figure(figsize=(5,5))
    my_circle=plt.Circle((0,0), 0.6, color='white')
    plt.pie(negative_word_counts['counts'], labels=negative_word_counts['words'], colors=Pastel1_7.hex_colors)
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    filename_top_20_neg = "top_20_neg_latest.png"
    plt.savefig(os.path.join(IMG_DIR, filename_top_20_neg), dpi = 300)
    plt.close()
    