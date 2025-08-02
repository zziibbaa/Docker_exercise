import mlflow
from mlflow import log_metric
from random import choice

#metric_name = ['CPU']
metric_name =['disk' , 'RAM' ,'cpu']
    
percentage = [ i for i in range (0 , 40)]
for i in range (40):
      log_metric(choice(metric_name) , choice(percentage))


