from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

dag = DAG("brain_tumor_pipeline", start_date=datetime(2026, 1, 1), schedule="@weekly")

t1 = BashOperator(task_id="check_data", bash_command="python src/data_loader.py", dag=dag)
t2 = BashOperator(task_id="validate_yolo", bash_command="python src/data_loader.py", dag=dag)
t3 = BashOperator(task_id="train_classifier", bash_command="python src/train.py", dag=dag)
t4 = BashOperator(task_id="train_detector", bash_command="python src/train.py", dag=dag)
t5 = BashOperator(task_id="evaluate", bash_command="python src/evaluate.py", dag=dag)
t6 = BashOperator(task_id="register", bash_command="python src/evaluate.py", dag=dag)

[t1, t2] >> t3
[t1, t2] >> t4
[t3, t4] >> t5 >> t6
