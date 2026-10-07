# Experimentación con MLflow

El script `train.py` entrena un SVC (con StandardScaler) para clasificar vinos, evalúa el modelo y registra cada corrida en MLflow.

## Requisitos necesarios:

- Python 3.14
- Y para instalar las dependencias, se recomienda usar el siguiente código:

pip install -r requirements.txt

## Instrucciones de ejecución

Con los valores por defecto, se puede correr el programa con este código:

python train.py


Y si se requiere cambiar hiperparámetros del mismo sin editar el código, se puede correr esta otra línea del código:

Ejemplo:

python train.py --kernel rbf --C 1 --gamma 1


Todos los argumentos disponibles: `--C`, `--kernel` (rbf, linear, poly, sigmoid), `--gamma`, `--test_size`, `--random_state` y `--cv_folds`. 

## Resultados en MlFlow

Para ver los diferentes experimentos y comprobar las métricas se puede visualizar mediante MLFlow, utilizando el siguiente código:


mlflow ui --backend-store-uri sqlite:///mlflow.db


Ya puesto esa línea de código, quedaría abrir la siguiente URL en tu navegador: http://127.0.0.1:5000, entrar al experimento `Vino_svc`, seleccionar las corridas y presionar *Compare*.

## Reproducibilidad

La semilla (`random_state` = 42) se usa en la partición train/test, en el SVC y en la validación cruzada.

Usando esta misma semilla se puede correr la siguiente línea de código y saldran los valores señalados:

python train.py --kernel rbf --C 1


 Ejecución | Accuracy (test) | CV accuracy (5-fold) 

 1         | 0.9722          | 0.9833 ± 0.0222 


Los resultados son idénticos, por lo que el experimento es reproducible.