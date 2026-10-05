import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)


def carga_datos():
    vinos = load_wine()
    X =pd.DataFrame(vinos.data,columns=vinos.feature_names)
    y =pd.Series(vinos.target, name="target")
    return X,y, vinos.target_names


def split_datos(X, y,test_size=0.20, random_state=42):
    Xtrain,Xtest,ytrain,ytest = train_test_split(
        X,y, test_size=test_size,random_state=random_state,stratify=y
    )
    return Xtrain, Xtest, ytrain, ytest


def pipeline_use(C=1.0, kernel="rbf",gamma="scale", random_state=42):
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", SVC(C=C,kernel=kernel,gamma=gamma,random_state=random_state))
    ])
    return pipeline


def entrenamiendo_modelo(pipeline, Xtrain, ytrain):
    pipeline.fit(Xtrain, ytrain)
    return pipeline
    
def evaluacion_modelo(pipeline, Xtest, ytest, target_names):
    y_pred = pipeline.predict(Xtest)

    metrics = {
        "accuracy": accuracy_score(ytest, y_pred),
        "precision_weighted": precision_score(ytest,y_pred, average="weighted"),
        "recall_weighted": recall_score(ytest,y_pred,average="weighted"),
        "f1_weighted": f1_score(ytest, y_pred,average="weighted"),
    }

    print("Resumen de clasificación:")
    print(classification_report(ytest, y_pred, target_names=target_names))

    return metrics, y_pred

def imagen_confusion(ytest, y_pred, target_names, output_path="confusion_matriz.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(ytest, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Matriz de confusión")
    plt.tight_layout()
    plt.savefig(output_path,dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def cross_val_acc(pipeline, X, y, cv_folds=5, random_state=42):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    return cv_scores


def parse_args():

    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelo SVC para clasificar vinos (Wine dataset)."
    )
    parser.add_argument(
        "--C", type=float, default=1.0,
        help="Parámetro de regularización del SVC"
    )
    parser.add_argument(
        "--kernel", type=str, default="rbf",
        choices=["rbf","linear","poly","sigmoid"],
        help="Kernel del SVC (default: rbf)"
    )
    parser.add_argument(
        "--gamma", type=str, default="scale",
        help="Coeficiente gamma del kernel:'scale', 'auto', o un número"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.20,
        help="Proporción del dataset para prueba"
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Semilla aleatoria, para reproducibilidad"
    )
    parser.add_argument(
        "--cv_folds", type=int, default=5,
        help="Número de folds para la validación cruzada"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gamma = args.gamma
    try:
        gamma = float(gamma)
    except ValueError:
        pass

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Vino_svc")

    print("Hiperparámetros de esta corrida:")
    print(f"C = {args.C}")
    print(f"kernel ={args.kernel}")
    print(f"gamma ={gamma}")
    print(f"test_size  = {args.test_size}")
    print(f"random_state= {args.random_state}")
    print(f"cv_folds = {args.cv_folds}")
    print()

    with mlflow.start_run():
        mlflow.log_param("C", args.C)
        mlflow.log_param("kernel", args.kernel)
        mlflow.log_param("gamma", gamma)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("cv_folds", args.cv_folds)

        X, y, target_names = carga_datos()
        Xtrain, Xtest, ytrain, ytest = split_datos(X, y,test_size=args.test_size,random_state=args.random_state)

        pipeline= pipeline_use(C=args.C, kernel=args.kernel,gamma=gamma,random_state=args.random_state)
        pipeline = entrenamiendo_modelo(pipeline, Xtrain, ytrain)
        mlflow.sklearn.log_model(pipeline, name="model")

        metrics,y_pred = evaluacion_modelo(pipeline,Xtest,ytest, target_names)
        print("\nMétricas en test:")
        for nombre, valor in metrics.items():
            print(f"  {nombre}: {valor:.4f}")
            mlflow.log_metric(nombre, valor)

        cm_path = imagen_confusion(ytest, y_pred, target_names)
        print(f"\nMatriz de confusión en: {cm_path}")
        mlflow.log_artifact(cm_path)

        cv_scores = cross_val_acc(pipeline,X,y, cv_folds=args.cv_folds, random_state=args.random_state)
        print(f"\nAccuracy promedio ({args.cv_folds}-fold CV): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std", cv_scores.std())

if __name__ == "__main__":
    main()
