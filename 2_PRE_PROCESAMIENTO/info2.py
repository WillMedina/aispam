import pandas as pd
import numpy as np
from collections import Counter
import re

def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(filepath, **kwargs)
    return df

def basic_overview(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("\nHead:")
    print(df.head())
    print("\nTail:")
    print(df.tail())
    print("\nData types:")
    print(df.dtypes)
    print("\nDescriptive statistics:")
    #print(df.describe(include='all', datetime_is_numeric=True))
    print(df.describe())

def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    total = df.isnull().sum()
    percent = (total / len(df)) * 100
    missing_df = pd.DataFrame({'total_missing': total, 'percent_missing': percent})
    return missing_df.sort_values(by='percent_missing', ascending=False)

def duplicate_rows_report(df: pd.DataFrame) -> pd.DataFrame:
    dup_mask = df.duplicated(keep='first')
    duplicates = df[dup_mask]
    return duplicates

def correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=['number'])
    corr_df = numeric_df.corr(method=method)
    return corr_df

def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    cats = df.select_dtypes(include=['object', 'category'])
    summary = []
    for col in cats.columns:
        counts = cats[col].value_counts(dropna=False)
        summary.append({
            'column': col,
            'unique_values': cats[col].nunique(dropna=True),
            'top': counts.idxmax(),
            'top_frequency': counts.max()
        })
    return pd.DataFrame(summary).set_index('column')

def save_dataframe(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    df.to_csv(filepath, index=False, **kwargs)

# Funciones específicas para dataset de mensajes y clasificación binaria

def class_distribution(df: pd.DataFrame) -> pd.Series:
    return df['tipo'].value_counts(normalize=True) * 100

def message_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    df['msg_length'] = df['mensaje'].apply(len)
    df['word_count'] = df['mensaje'].apply(lambda x: len(x.split()))
    return df[['msg_length', 'word_count']].describe()

# Esta funcion esta devolviendo vacios no sé por que
def most_common_words(df: pd.DataFrame, n=10) -> dict:
    spam_words = Counter()
    ham_words = Counter()
    for _, row in df.iterrows():
        words = re.findall(r"\\b\\w+\\b", row['mensaje'].lower())
        if row['tipo'].lower() == 'spam':
            spam_words.update(words)
        else:
            ham_words.update(words)
    return {
        'spam': spam_words.most_common(n),
        'ham': ham_words.most_common(n)
    }

def average_words_per_class(df: pd.DataFrame) -> pd.DataFrame:
    df['word_count'] = df['mensaje'].apply(lambda x: len(x.split()))
    return df.groupby('tipo')['word_count'].mean()

if __name__ == "__main__":
    CSV_PATH = '../DATASETS_GENERADOS/dataset_spam_ham_flax-community_gpt-2-spanish_15000_48.csv'
    df = load_csv(CSV_PATH)
    
    print("\n=== Valores faltantes ===")
    print(missing_values_report(df))
    
    df = df.dropna(subset=['mensaje', 'tipo']) #quitando los nulos
    
    print("=== Visión básica ===")
    basic_overview(df)

    
    print("\n=== Duplicados ===")
    print(duplicate_rows_report(df))

    print("\n=== Categorías ===")
    print(categorical_summary(df))

    print("\n=== Distribución de clases (tipo) ===")
    print(class_distribution(df))

    print("\n=== Estadísticas de longitud de mensaje ===")
    print(message_length_stats(df))

    print("\n=== Palabras más comunes por clase ===")
    common_words = most_common_words(df)
    print("Spam:", common_words['spam'])
    print("Ham:", common_words['ham'])

    print("\n=== Promedio de palabras por mensaje por clase ===")
    print(average_words_per_class(df))
