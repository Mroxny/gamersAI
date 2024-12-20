from kedro.framework.hooks import hook_impl
import wandb
import os

class WAndBPipelineHook:
    def __init__(self, project_name):
        self.project_name = project_name

    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog, **kwargs):
        # Initialize W&B once at the start of the pipeline
        os.environ["WANDB_RUN_GROUP"] = "experiment- " + wandb.util.generate_id()

