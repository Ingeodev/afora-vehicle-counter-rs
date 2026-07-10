use crate::core::afora_error::AforaError;
use crate::features::pipeline::adapters::multithreaded_pipeline::MultithreadedPipeline;
use crate::features::pipeline::adapters::sequential_pipeline::SequentialPipeline;
use crate::features::pipeline::domain::pipeline_config::{ExecutionMode, PipelineConfig};
use crate::features::pipeline::ports::pipeline::Pipeline;

pub struct PipelineFactory {
    pipeline_config: PipelineConfig
}

impl PipelineFactory {
    pub fn build(pipeline_config: PipelineConfig) -> Result<Box<dyn Pipeline>, AforaError> {
        match pipeline_config.execution_mode {
            ExecutionMode::Sequential =>
                Ok(Box::new(
                    SequentialPipeline::new(
                        pipeline_config.media_source,
                        pipeline_config.detector,
                        pipeline_config.tracker,
                        pipeline_config.subscribers,
                    )
                )),
            ExecutionMode::Multithreaded =>
                Ok(Box::new(
                    MultithreadedPipeline::new(
                        pipeline_config.media_source,
                        pipeline_config.detector,
                        pipeline_config.tracker,
                        pipeline_config.subscribers,
                    )
                )),
        }
    }
}