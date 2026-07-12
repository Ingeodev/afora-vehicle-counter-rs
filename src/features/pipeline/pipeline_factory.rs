use crate::core::afora_error::AforaError;
use crate::features::pipeline::adapters::multithreaded_pipeline::MultithreadedPipeline;
use crate::features::pipeline::adapters::sequential_pipeline::SequentialPipeline;
use crate::features::pipeline::adapters::threated_subscriber_broadcaster::ThreadedSubscriberBroadcaster;
use crate::features::pipeline::domain::pipeline_config::{ExecutionMode, PipelineConfig};
use crate::features::pipeline::ports::pipeline::Pipeline;
use crate::features::pipeline::ports::subscriber_broadcast::SubscriberBuilder;
use crate::features::tracking_suscribers::tracking_subscriber_factory::{TrackerSubscriberChoice, TrackerSubscriberFactory};

pub struct PipelineFactory {
    pipeline_config: PipelineConfig
}

impl PipelineFactory {
    pub fn build(pipeline_config: PipelineConfig) -> Result<Box<dyn Pipeline>, AforaError> {

        let broadcaster = Box::new(Self::build_broadcaster(pipeline_config.subscribers));

        match pipeline_config.execution_mode {
            ExecutionMode::Sequential =>
                Ok(Box::new(
                    SequentialPipeline::new(
                        pipeline_config.media_source,
                        pipeline_config.detector,
                        pipeline_config.tracker,
                        broadcaster,
                    )
                )),
            ExecutionMode::Multithreaded =>
                Ok(Box::new(
                    MultithreadedPipeline::new(
                        pipeline_config.media_source,
                        pipeline_config.detector,
                        pipeline_config.tracker,
                        broadcaster,
                    )
                )),
        }
    }

    fn build_broadcaster(
        subscriber_choices: Vec<TrackerSubscriberChoice>,
    ) -> ThreadedSubscriberBroadcaster {
        let builders: Vec<SubscriberBuilder> = subscriber_choices
            .into_iter()
            .map(|choice| -> SubscriberBuilder {
                Box::new(move || TrackerSubscriberFactory::build(choice))
            })
            .collect();

        ThreadedSubscriberBroadcaster::new(builders)
    }
}