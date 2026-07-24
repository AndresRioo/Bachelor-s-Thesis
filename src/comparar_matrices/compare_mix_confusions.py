import argparse
import pandas as pd
import numpy as np


def load_mix_matrix(path):
    """
    Carga la matriz de confusión agregada desde un archivo Excel.

    Args:
        path (str):
            Ruta al archivo Excel que contiene la hoja
            'Confusion_Mix'.

    Returns:
        pd.DataFrame:
            Matriz de confusión con las clases como índices
            y columnas.
    """
    df = pd.read_excel(path, sheet_name="Confusion_Mix", index_col=0)
    return df



def compute_top_confusions(cm, top_k=50):
    """
    Obtiene las confusiones más frecuentes excluyendo la
    diagonal principal.

    Args:
        cm (pd.DataFrame):
            Matriz de confusión.

        top_k (int):
            Número máximo de confusiones a devolver.

    Returns:
        pd.DataFrame:
            DataFrame ordenado por frecuencia de confusión
            descendente.
    """
    classes = cm.index.tolist()
    confusions = []

    for i in range(len(classes)):
        for j in range(len(classes)):

            if i == j:
                continue

            value = cm.iloc[i, j]

            if value > 0:
                confusions.append(
                    (classes[i], classes[j], value)
                )

    df = pd.DataFrame(
        confusions,
        columns=["True_Class", "Predicted_Class", "Count"]
    )

    df = df.sort_values("Count", ascending=False)

    return df.head(top_k)


def compute_confusion_changes(diff_matrix, top_k=50):
    """
    Identifica las confusiones que más aumentan y más disminuyen
    entre dos matrices de confusión.

    Args:
        diff_matrix (pd.DataFrame):
            Diferencia entre dos matrices de confusión
            (matriz_B - matriz_A).

        top_k (int):
            Número máximo de cambios a devolver en cada sentido.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            DataFrames con los mayores incrementos y las mayores
            reducciones de confusión.
    """
    classes = diff_matrix.index.tolist()
    changes = []

    for i in range(len(classes)):
        for j in range(len(classes)):

            if i == j:
                continue

            value = diff_matrix.iloc[i, j]

            if value != 0:
                changes.append(
                    (classes[i], classes[j], value)
                )

    df = pd.DataFrame(
        changes,
        columns=["True_Class", "Predicted_Class", "Change"]
    )

    df_increase = df.sort_values("Change", ascending=False).head(top_k)
    df_decrease = df.sort_values("Change", ascending=True).head(top_k)

    return df_increase, df_decrease


def compute_class_accuracy(cm):
    """
    Calcula la accuracy por clase a partir de una matriz
    de confusión.

    Args:
        cm (pd.DataFrame):
            Matriz de confusión.

    Returns:
        pd.DataFrame:
            DataFrame con el nombre de cada clase y su accuracy.
    """
    acc = cm.values.diagonal() / np.maximum(cm.sum(axis=1).values, 1)

    df = pd.DataFrame({
        "class": cm.index,
        "accuracy": acc
    })

    return df



def main():
    """
    Compara dos matrices de confusión agregadas y genera un
    informe Excel con estadísticas de diferencias, confusiones
    más frecuentes y variación de accuracy por clase.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--excel_a", required=True)
    parser.add_argument("--excel_b", required=True)
    parser.add_argument("--output", default="mix_comparison.xlsx")

    args = parser.parse_args()

    print("\nLoading matrices...")

    cm_a = load_mix_matrix(args.excel_a)
    cm_b = load_mix_matrix(args.excel_b)

    if not cm_a.index.equals(cm_b.index):
        raise ValueError("Class labels do not match between excels")

    # MATRIX DIFFERENCE

    diff = cm_b - cm_a

    # TOP CONFUSIONS

    top_a = compute_top_confusions(cm_a)
    top_b = compute_top_confusions(cm_b)

    # CONFUSION CHANGES

    increase, decrease = compute_confusion_changes(diff)

    # CLASS ACCURACY
    acc_a = compute_class_accuracy(cm_a)
    acc_b = compute_class_accuracy(cm_b)

    df_gain = acc_a.copy()
    df_gain.rename(columns={"accuracy": "accuracy_A"}, inplace=True)

    df_gain["accuracy_B"] = acc_b["accuracy"]
    df_gain["gain"] = df_gain["accuracy_B"] - df_gain["accuracy_A"]

    df_gain = df_gain.sort_values("gain", ascending=False)

    # SAVE EXCEL
    print("Saving comparison Excel...")

    with pd.ExcelWriter(args.output) as writer:

        cm_a.to_excel(writer, sheet_name="Mix_A")
        cm_b.to_excel(writer, sheet_name="Mix_B")

        diff.to_excel(writer, sheet_name="Mix_difference")

        top_a.to_excel(writer, sheet_name="Top_confusions_A", index=False)
        top_b.to_excel(writer, sheet_name="Top_confusions_B", index=False)

        increase.to_excel(writer, sheet_name="Confusions_increased", index=False)
        decrease.to_excel(writer, sheet_name="Confusions_decreased", index=False)

        df_gain.to_excel(writer, sheet_name="Class_accuracy_gain", index=False)

    print("\nDone.")
    print("Output file:", args.output)



if __name__ == "__main__":
    main()