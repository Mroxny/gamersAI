from kedro.framework.hooks import hook_impl
import wandb

class WAndBPipelineHook:
    def __init__(self, project_name):
        self.project_name = project_name

    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog, **kwargs):
        
        # Initialize W&B once at the start of the pipeline
        wandb.init(
            project=self.project_name
        )

    @hook_impl
    def after_pipeline_run(self, run_params, pipeline, catalog, **kwargs):
        # Finalize W&B after the pipeline finishes
        wandb.finish()