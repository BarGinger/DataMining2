"""
File: analysis.py
Student 1: Bar Melinarskiy, student number: 2482975
Student 2: Prathik Alex Matthai, student number: 9020675
Student 3: Mohammed Bashabeeb, student number: 7060424
Date: September 12, 2024
Description: Assignment 2 - Classification for the Detection of Opinion Spam - analysis for report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.metrics import confusion_matrix
import seaborn as sns
from wordcloud import WordCloud

from main import get_df, FAKE, EVALUATIONS_FILENAME


def create_word_cloud(words: str, title: str):
    """
    Plot word cloud on given dataframe

    Parameters:
    -----------
    df : str a string of list of words
    title : str, title for the word cloud plot

    Returns:
    --------
    None
    """
    # Create and configure word cloud
    wordcloud = WordCloud( width=1000,
                           height=1000,
                           background_color='white',
                           colormap='magma',
                           max_words=5000,
                           random_state=42).generate(words)

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.clf()
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(title, fontsize=12, fontweight='bold', loc='center')
    plt.savefig(f"../Output/{title}.png")
    plt.show()



def plot_word_clouds():
    """
    Plot word cloud on entire fake reviews text, and on top 5 words from the different models
    """

    ## plot word cloud for entire fake review dataset
    df = get_df()
    fake_reviews = df[df['is_fake'] == FAKE.DECEPTIVE]['review']
    words = " ".join(review.lower() for review in fake_reviews.dropna())
    create_word_cloud(words, title="Most frequent words in fake reviews dataset")

    ## plot word cloud for entire truthful review dataset
    df = get_df()
    truthful_reviews = df[df['is_fake'] == FAKE.TRUTHFUL]['review']
    truthful_words = " ".join(review.lower() for review in truthful_reviews.dropna())
    create_word_cloud(truthful_words, title="Most frequent words in truthful reviews dataset")

    ## plot word cloud for top 5 words from models
    df_evaluations = pd.read_csv(EVALUATIONS_FILENAME)
    top_5_features_deceptive = df_evaluations['top_5_features_deceptive']
    top_5_words = " ".join(
        word.strip() for top_words in top_5_features_deceptive.dropna() for word in top_words.split(','))
    create_word_cloud(top_5_words, title="Top 5 terms relating to fake reviews across all models")


def plot_confusion_matrix(y_true, y_pred, title):
    """
    Plot word cloud on entire fake reviews text, and on top 5 words from the different models
    """
    confusion_matrix_data = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cm_len = len(confusion_matrix_data)
    df_cm = pd.DataFrame(confusion_matrix_data, index=range(0, cm_len), columns=range(0, cm_len))
    group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    group_counts = ["{0: 0.0f}".format(value) for value in confusion_matrix_data.flatten()]
    group_percentages = ['{0:.2%}'.format(value)
                         for value in confusion_matrix_data.flatten() / np.sum(confusion_matrix_data)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.set(font_scale=1.2)  # Increase font size
    sns.heatmap(df_cm, annot=labels, fmt='', cmap='Blues', cbar=False,
                annot_kws={"size": 14, "fontweight": "bold"})  # Adjust font properties
    plt.title(title, fontsize=16, fontweight='bold')  # Make title bold and larger
    file_name = f"Output/{title}_confusion_matrix.png"
    plt.savefig(file_name, dpi=450)
    plt.show()

def plot_scores_as_bars(df_scores):
    """
    Plot word cloud on given dataframe

    Parameters:
    -----------
    ddf_scores : dataframe of scores

    Returns:
    --------
    None
    """
    # Melt the DataFrame to long format
    df_melted = df_scores.melt(id_vars='model_name', var_name='Metric', value_name='Value')

    df_melted.to_csv("Output/df_melted.csv", sep='\t', encoding='utf-8')

    # Create the bar plot
    plt.figure(figsize=(20, 16))
    ax = sns.barplot(x='Metric', y='Value', hue='Model', data=df_melted)

    # Add value labels on the bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 9),
                    textcoords='offset points'
                    )

    plt.title(label='Comparison of performance metrics across the different models',
              fontsize=34,
              fontweight='bold',
              color='brown',
              pad=20)

    plt.ylabel("Score", fontsize=30, fontweight='bold', color='black', labelpad=20)
    plt.xlabel("Metrics", fontsize=30, fontweight='bold', color='black', labelpad=20)

    # Increase size of axis ticks and values
    plt.xticks(fontsize=16, color='navy', fontweight='bold', rotation=45)
    plt.yticks(fontsize=16, color='navy', fontweight='bold')
    plt.ylim(0, 1)
    plt.legend(title='Model')

    plt.subplots_adjust(top=0.85)
    file_name = "Output/Comparison of performance Metrics.png"
    plt.savefig(file_name, dpi=450)
    plt.show()


def plot_scores_as_lines(df_scores):
    """
    Plot scores as a multi-line plot on given dataframe

    Parameters:
    -----------
    df_scores : dataframe of scores

    Returns:
    --------
    None
    """

    df_mnb = df_scores[df_scores['model_name'].str.startswith('Multinomial Naive Bayes')]
    df_mnb['feature_count'] = df_mnb['model_name'].str.extract(r'\(#(\d+) features\)')

    columns_to_keep = ['feature_count', 'dataset_name', 'accuracy', 'f1', 'precision', 'recall']

    df_mnb = df_mnb.loc[:, columns_to_keep]

    # Melt the DataFrame to long format
    df_melted = df_mnb.melt(id_vars=['feature_count', 'dataset_name'], var_name='metric', value_name='value')

    # df_melted.to_csv("Output/df_melted.csv", sep='\t', encoding='utf-8')

    # Create the line plot
    plt.figure(figsize=(20, 16))
    # Create a FacetGrid for the plot
    g = sns.FacetGrid(df_melted, col='dataset_name', hue='metric', sharey=False, height=5, aspect=1.5)
    g.map(sns.lineplot, 'feature_count', 'value')

    # Add titles and labels
    g.set_axis_labels("Feature Count", "Score", fontsize=16)
    g.set_titles("{col_name}", size=18)
    g.add_legend(title="Metrics", title_fontsize=20, fontsize=15)
    g.fig.suptitle("Model Performance Metrics by Feature Count and Dataset", fontsize=20, y=1.0002)

    legend = g.legend
    legend.set_title("Metrics")
    legend.get_title().set_fontsize(16)  # Legend title font size
    for text in legend.get_texts():
        text.set_fontsize(14)  # Legend label font size


    # Increase the size of axis tick labels
    # Increase the size of axis tick labels without triggering the warning
    g.set_xticklabels(fontsize=12)
    g.set_yticklabels(fontsize=12)


    plt.subplots_adjust(top=0.8)
    file_name = "../Output/Comparison of performance Metrics.png"
    plt.savefig(file_name, dpi=450)
    plt.show()



if __name__ == "__main__":
    # plot_word_clouds()

    df_evaluations = pd.read_csv(EVALUATIONS_FILENAME)
    plot_scores_as_lines(df_evaluations)


