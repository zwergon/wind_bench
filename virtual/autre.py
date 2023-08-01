from clearml import Task
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, RandomSearch,
    UniformIntegerParameterRange, UniformParameterRange)
task = Task.init(project_name='ai.virtual',
                 task_name='Automatic Hyper-Parameter Optimization',
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)

an_optimizer = HyperParameterOptimizer( base_task_id="9db60379db924c20a9ce03ae37d6d004", 
                                        hyper_parameters=[
          DiscreteParameterRange('General/epoch',values=[50, 100]),
           DiscreteParameterRange('General/batch_size', values=[200, 300, 400]),
          #UniformParameterRange('General/learning_rate', min_value=0.00001, max_value=0.1, step_size=0.01),
          ], objective_metric_title='train_loss',
    objective_metric_series='train_loss', objective_metric_sign='min')

an_optimizer.start()
an_optimizer.wait()
an_optimizer.stop()

