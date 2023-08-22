import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
import zipfile
import os
import sys

def parse_data(data_path):

    data_folder = 'data'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    with zipfile.ZipFile(data_path, 'r') as zip_ref:
        zip_ref.extractall(data_folder)

    csv_files = [file for file in os.listdir(data_folder) if file.endswith('.csv')]

    dataframes = {}
    for csv_file in csv_files:
        df_name = os.path.splitext(csv_file)[0]
        csv_file_path = os.path.join(data_folder, csv_file)
        dataframes[df_name] = pd.read_csv(csv_file_path)
    return dataframes

def preprocessing(dataframes):

    merged_test = pd.merge(dataframes['NBKI_test'], dataframes['NBKI_y_test'], on='Unnamed: 0')
    merged_test['split'] = 'test'
    full_data = pd.concat([dataframes['NBKI_train'], merged_test], ignore_index=True)
    full_data['split'].fillna('train', inplace=True)
    full_data.rename(columns={'Unnamed: 0': 'key_val'}, inplace=True)
    full_data['default'] = full_data['default'].astype(int)

    missing_values = full_data.isnull().sum()

    columns_with_missing = missing_values[missing_values > 0].index.tolist()

    imputer = SimpleImputer(strategy='most_frequent')  # замена пропусков на моду
    imputed_data = full_data.copy()
    imputed_data[columns_with_missing] = imputer.fit_transform(full_data[columns_with_missing])

    return imputed_data

def split_data(data):
    shuffled_df = data.sample(frac=1, random_state=42)
    train_data = shuffled_df[shuffled_df['split'] == 'train']
    test_data = shuffled_df[shuffled_df['split'] == 'test']
    target_column = 'default'
    X_train = train_data.drop(columns=['split', target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=['split', target_column])
    y_test = test_data[target_column]

    return X_train, X_test, y_test, y_train


def get_metrics(y_test, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)  # Прогнозы на основе вероятностей с заданным порогом

    auc_score = roc_auc_score(y_test, y_probs)
    gini_auc = 2 * auc_score - 1

    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    precision_0 = classification_rep['0']['precision']
    recall_0 = classification_rep['0']['recall']
    f1_0 = classification_rep['0']['f1-score']
    support_0 = classification_rep['0']['support']

    precision_1 = classification_rep['1']['precision']
    recall_1 = classification_rep['1']['recall']
    f1_1 = classification_rep['1']['f1-score']
    support_1 = classification_rep['1']['support']

    metrics_text = f"\tGini (AUC): {gini_auc:.4f}\n\n\tClass 0:\n\tPrecision: {precision_0:.4f}\n\t\Recall: {recall_0:.4f}\n\tF1-Score: {f1_0:.4f}\n\tSupport: {support_0}\n\n\tClass 1:\n\tPrecision: {precision_1:.4f}\n\tRecall: {recall_1:.4f}\n\tF1-Score: {f1_1:.4f}\n\tSupport: {support_1}"

    print(metrics_text)

def validate(X_test, y_test, model, threshold=0.5, feature_to_analyze=None):
    y_probs = model.predict_proba(X_test)[:, 1]
    check = {}
    print("\t\t\tOverall Model Evaluation")
    get_metrics(y_test, y_probs, threshold)

    if feature_to_analyze:
        unique_values = X_test[feature_to_analyze].unique()
        for value in unique_values:
            idx = X_test[feature_to_analyze] == value
            print(f"\n\t\t Analysis by Split of Feature '{feature_to_analyze}' = {value}")
            get_metrics(y_test[idx], y_probs[idx], threshold)
            check[value] = (y_test[idx], y_probs[idx])
    return check


def modeling(data):
    X_train, X_test, y_test, y_train = split_data(data)
    pos_class_count = y_train.sum()
    neg_class_count = len(y_train) - pos_class_count
    scale_pos_weight = neg_class_count / pos_class_count  # Это отношение показывает, насколько <важен> положительный класс
    # относительно отрицательного класса для обучения модели.

    model_cb = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.1, loss_function='Logloss', verbose=False,
                                  scale_pos_weight=scale_pos_weight)
    model_cb.fit(X_train, y_train)

    test_check = validate(X_test, y_test, model_cb, threshold=0.3, feature_to_analyze='0')  # use threshold=0.3
    return test_check


def logg_(df):
    output_file = 'predictions.csv'
    for split in df.keys():
        result_df = pd.DataFrame({'y_test': test_check[split][0], 'y_probs': test_check[split][1], 'split': split})
    result_df.to_csv(output_file, index=False)

    print(f"Predictions saved to {output_file}")


if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise ValueError("Usage: python script.py <input_csv_file>")

    dfs = parse_data(sys.argv[1])
    prepro_data = preprocessing(dfs)
    test_check = modeling(prepro_data)
    logg_(test_check)
