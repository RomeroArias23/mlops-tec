# MLOps — Tecnológico de Monterrey

Repositorio de trabajo para el curso de **MLOps** de la Maestría en Inteligencia Artificial Aplicada.

# El script train.py permite:
- Ejecutar el entrenamiento desde la línea de comandos.
- Configurar hiperparámetros mediante argumentos CLI.
- Utilizar una semilla aleatoria para reproducibilidad.
- Registrar experimentos mediante MLflow.
- Registrar parámetros y métricas.
- Generar artefactos de evaluación.
- Registrar el modelo entrenado.
- Comparar diferentes configuraciones dentro de un mismo experimento.

## Estructura

```text
mlops-tec/
├── README.md
├── requirements.txt
│
├── week-01/
│   ├── 01-paper-analysis/
│   ├── 02-stack-setup/
│   └── 03-hello-mlflow/
│       ├── hello_mlflow.py
│       └── hello_artifact.txt
│
├── week-02/
│
├── week-03/
│   ├── Attribute DataSet.xlsx
│   └── baseline-model.ipynb
│
└── week-04/
    ├── Attribute DataSet.xlsx
    ├── train.py
    └── artifacts/
        ├── classification_report.txt
        └── confusion_matrix.txt
