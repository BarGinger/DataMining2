"""
File: analysis.py
Student 1: Bar Melinarskiy, student number: 2482975
Student 2: Prathik Alex Matthai, student number: 9020675
Student 3: Mohammed Bashabeeb, student number: 7060424
Date: September 12, 2024
Description: Assignment 2 - Classification for the Detection of Opinion Spam - analysis for report
"""
from fileinput import filename

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.metrics import confusion_matrix
import seaborn as sns
from wordcloud import WordCloud
from utils import get_df, FAKE, EVALUATIONS_FILENAME


def create_word_cloud(words: str, filename: str, title: str, colored_word=None, color=None):
    """
    Plot word cloud with an optional colored word in the title.

    Parameters:
    -----------
    words : str
        A string of words to generate the word cloud.
    filename : str
        Filename to save the plot.
    title : str
        Title for the word cloud plot.
    colored_word : str, optional
        Word to color differently in the title.
    color : str, optional
        Color of the colored_word.

    Returns:
    --------
    None
    """
    # Create and configure word cloud
    wordcloud = WordCloud(width=1000,
                          height=1000,
                          background_color='white',
                          colormap='magma',
                          max_words=5000,
                          random_state=42).generate(words)

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')

    # Display title with colored word
    if colored_word and color:
        # Split title around colored word
        title_parts = title.split(colored_word)


        x = [0.55, 0.6, 0.655]
        if colored_word == 'truthful':
            x = [0.5, 0.6, 0.705]

        if 'Top' in title:
            x = [x_i + 0.14 for x_i in x]


        # Plot the first part of the title
        plt.text(x[0], 1.05, title_parts[0], fontsize=14, fontweight='bold', ha='right', transform=plt.gca().transAxes)

        # Plot the colored word in the middle
        plt.text(x[1], 1.05, colored_word, fontsize=14, fontweight='bold', color=color, ha='center',
                 transform=plt.gca().transAxes)

        # Plot the rest of the title after the colored word
        plt.text(x[2], 1.05, title_parts[1], fontsize=14, fontweight='bold', ha='left', transform=plt.gca().transAxes)
    else:
        plt.title(title, fontsize=14, fontweight='bold')

    # Save and show the plot
    plt.savefig(f"../Output/{filename}.png")
    plt.show()


def plot_word_clouds():
    """
    Plot word cloud on entire fake reviews text, and on top 5 words from the different models.
    """
    # Load dataset
    df = get_df()

    # Plot word cloud for entire fake review dataset
    fake_reviews = df[df['is_fake'] == FAKE.DECEPTIVE]['review']
    words = " ".join(review.lower() for review in fake_reviews.dropna())
    create_word_cloud(words, filename='word_cloud_fake',
                      title="Most frequent words in fake reviews dataset", colored_word="fake", color="red")

    # Plot word cloud for entire truthful review dataset
    truthful_reviews = df[df['is_fake'] == FAKE.TRUTHFUL]['review']
    truthful_words = " ".join(review.lower() for review in truthful_reviews.dropna())
    create_word_cloud(truthful_words, filename='word_cloud_truthful',
                      title="Most frequent words in truthful reviews dataset", colored_word="truthful", color="green")

    # Plot word cloud for top 5 words from models
    df_top_k_features = pd.read_csv("../Output/top_k_features.csv")

    filtered_values_deceptive = np.where((df_top_k_features['class'] ==  1))
    top_5_features_deceptive = np.array(df_top_k_features.loc[filtered_values_deceptive]['feature'])
    top_5_features_deceptive_str = ' '.join(top_5_features_deceptive)
    create_word_cloud(top_5_features_deceptive_str, filename='word_cloud_top_deceptive',
                      title="Top features predicting fake reviews",
                      colored_word="fake", color="red")


    filtered_values_truthful = np.where((df_top_k_features['class'] == 0))
    top_5_features_truthful = np.array(df_top_k_features.loc[filtered_values_truthful]['feature'])
    top_5_features_truthful_str = ' '.join(top_5_features_truthful)
    create_word_cloud(top_5_features_truthful_str, filename='word_cloud_top_truthful',
                      title="Top features predicting truthful reviews",
                      colored_word="truthful", color="green")



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


def plot_bnm_scores_as_lines(df_scores):
    """
    Plot scores as a multi-line plot on given dataframe

    Parameters:
    -----------
    df_scores : dataframe of scores

    Returns:
    --------
    None
    """

    df_mnb = df_scores[df_scores['model_name'].str.startswith('Multinomial Naive Bayes')].copy()
    df_mnb['feature_count'] = df_mnb['model_name'].str.extract(r'\(#(\d+) features\)')

    columns_to_keep = ['feature_count', 'dataset_name', 'accuracy', 'f1', 'precision', 'recall']

    df_mnb = df_mnb.loc[:, columns_to_keep]

    # Melt the DataFrame to long format
    df_melted = df_mnb.melt(id_vars=['feature_count', 'dataset_name'], var_name='metric', value_name='value')

    # df_melted.to_csv("Output/df_melted.csv", sep='\t', encoding='utf-8')

    # Create the line plot
    plt.figure(figsize=(4.54,4.54), dpi=800)
    # Create a FacetGrid for the plot
    g = sns.FacetGrid(df_melted, col='dataset_name', hue='metric', sharex=False, sharey=False, height=5, aspect=1.5)
    g.map(sns.lineplot, 'feature_count', 'value')

    # Add titles and labels
    g.set_axis_labels("Feature Count", "Score", fontsize=17.5)
    g.set_titles("{col_name}", size=25)
    g.add_legend(title="Metrics", title_fontsize=28, fontsize=20)
    g.fig.suptitle("Evaluating Multinomial Naive Bayes: Performance Metrics Based on Feature Count and N-Gram Selection",
                   fontsize=26,
                   y=1.0002,
                   color='navy',
                   fontweight='bold')

    legend = g.legend
    legend.set_title("Metrics")
    legend.get_title().set_fontsize(25)  # Legend title font size
    for text in legend.get_texts():
        text.set_fontsize(22)  # Legend label font size
    g.set_xticklabels(fontsize=14)
    g.set_yticklabels(fontsize=16)


    plt.subplots_adjust(top=0.8)
    file_name = "../Output/Multinomial Naive Bayes performance Metrics.png"
    plt.savefig(file_name, dpi=800)
    plt.show()



if __name__ == "__main__":
    df_evaluations = pd.read_csv(EVALUATIONS_FILENAME)
    # Step 1: Remove the closing parenthesis ')' from 'model_name'
    df_evaluations['model_name'] = df_evaluations['model_name'].str.replace(')', '', regex=False)

    # Step 2: Split 'model_name' into 'model_name' and 'settings' based on '('
    df_evaluations[['model_name', 'settings']] = df_evaluations['model_name'].str.split('(', expand=True)

    # Step 3: Strip whitespace from both 'model_name' and 'settings' columns
    df_evaluations['model_name'] = df_evaluations['model_name'].str.strip()
    df_evaluations['settings'] = df_evaluations['settings'].str.strip()

    # Group by 'model_name' and 'dataset_name', then find the row index with the maximum accuracy for each group
    max_accuracy_rows = df_evaluations.loc[df_evaluations.groupby(['model_name', 'dataset_name'])['accuracy'].idxmax()]


    # # create word cloud from entire dataset and from top features in LR
    # plot_word_clouds()
    # # create plot for comparing different n-grams, and # of features in Multinomial Naive Bayes
    # plot_bnm_scores_as_lines(df_evaluations)

    print("Done with plots")


