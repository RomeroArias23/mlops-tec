import mlflow
from pathlib import Path


# 1. Servidor de MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# 2. Experimento
mlflow.set_experiment("Hello World MLflow")


# 3. Crear un artifact
artifact_path = Path("hello_artifact.txt")

artifact_path.write_text(
    "Este archivo fue generado durante el experimento Hello World de MLflow.\n",
    encoding="utf-8"
)


# 4. Iniciar un Run
with mlflow.start_run():

    # 5. Registrar parámetros
    mlflow.log_param("experiment_type", "hello_world")
    mlflow.log_param("message_length", len("Hello MLflow"))


    # 6. Registrar métricas
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("execution_score", 1.0)


    # 7. Registrar el artifact
    mlflow.log_artifact(str(artifact_path))


    # 8. Mostrar información del Run
    run_id = mlflow.active_run().info.run_id

    print(f"Run ID: {run_id}")
    print("Experimento registrado correctamente en MLflow.")
