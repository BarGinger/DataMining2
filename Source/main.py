"""
File: main.py
Student 1: Bar Melinarskiy, student number: 2482975
Student 2: Prathik Alex Matthai, student number: 9020675
Student 3: Mohammed Bashabeeb, student number: 7060424
Date: September 12, 2024
Description: Assignment 2 - Classification for the Detection of Opinion Spam
"""

import zipfile

import numpy as np
import pandas as pd
from io import TextIOWrapper
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt


def read_csv_from_zip(zip_file_path: str) -> pd.DataFrame:
    """
        Read the csv of the data from the given zip file path
        @param zip_file_path: path to the zip file
        @return a dataframe of the given zip content
        """
    dfs = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        # Iterate over each file in the zip archive
        for file_info in zip_file.infolist():
            if file_info.filename.endswith('.csv'):  # Check if file is CSV
                # Open the file within the zip archive
                with zip_file.open(file_info.filename) as csv_file:
                    # Convert the binary stream to text and then read it with pandas
                    text_stream = TextIOWrapper(csv_file, encoding='utf-8')
                    # Specify the delimiter as ';'
                    df = pd.read_csv(text_stream, sep=';')
                    # Add a new column with the filename
                    df['__filename__'] = file_info.filename
                    dfs.append(df.reset_index(drop=True))
    # Concatenate all DataFrames into a single DataFrame
    result_df = pd.concat(dfs)
    return result_df


def compute_metrics(y_true, y_pred, title=""):
    """
        Compute the quality metrics for given predicted value compared to true labels.
        @param y_true: true labels
        @param y_pred: predicted values
        @param title: title for file to save results
        @:return the calculated metrics
        """
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Precision
    precision = precision_score(y_true, y_pred)

    # Recall
    recall = recall_score(y_true, y_pred)

    # F1 Score
    f1 = f1_score(y_true, y_pred)

    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    confusion_matrix_data = confusion_matrix(y_true=y_true, y_pred=y_pred)

    specificity = confusion_matrix_data[0, 0] / (confusion_matrix_data[0, 0] + confusion_matrix_data[0, 1])

    # df_metrics = classification_report(y_true=y_true, y_pred=y_pred)

    df_metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc
    }
    df_metrics = pd.DataFrame([df_metrics])
    result_txt = f"Output/{title}_results.txt"
    plot_title = f"Confusion Matrix for {title}"

    with open(result_txt, "w") as file:
        file.write(f"Requested quality measures results for - {title}: \n")
        file.write(tabulate(df_metrics, headers='keys', tablefmt='psql'))
        file.write("\n" + plot_title + ":\n")
        np.savetxt(file, confusion_matrix_data, fmt='%d')

    print(f"Requested quality measures results for - {title}: \n")
    print(tabulate(df_metrics, headers='keys', tablefmt='psql'))

    print(f"Requested quality measures results for - {title}: \n")
    print(tabulate(df_metrics, headers='keys', tablefmt='psql'))
    print(plot_title + ":")
    print(confusion_matrix_data)
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
    plt.title(plot_title, fontsize=16, fontweight='bold')  # Make title bold and larger
    file_name = f"Output/{title}_confusion_matrix.png"
    plt.savefig(file_name, dpi=450)
    plt.show()

    return df_metrics, df_cm


if __name__ == "__main__":
    df = read_csv_from_zip("Data/op_spam_v1.4.zip")
    # df_train = df[df['__filename__'] == 'eclipse-metrics-packages-2.0.csv']
    # X_train, Y_train, columns_train = get_x_y(df_train)
    # df_test = df[df['__filename__'] == 'eclipse-metrics-packages-3.0.csv']
    # X_test, Y_test, columns_test = get_x_y(df_test)

    print("Done!")
