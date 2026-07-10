use crate::core::afora_error::AforaError;
use crate::features::detector::{DetectorFactory, ModelChoice, RuntimeChoice};
use crate::features::media_source::domain::frame_source::FrameSource;
use crate::features::media_source::media_source_factory::{MediaSourceChoice, MediaSourceFactory};
use crate::features::pipeline::domain::pipeline_config::{ExecutionMode, PipelineConfig};
use crate::features::pipeline::pipeline_factory::PipelineFactory;
use crate::features::pipeline::ports::pipeline::Pipeline;
use crate::features::tracker::ports::tracker::Tracker;
use crate::features::tracker::tracker_factory::{TrackerChoice, TrackerFactory};
use crate::features::tracking_suscribers::ports::tracking_subscriber::TrackingSubscriber;

pub struct PipelineBuilder {
    pub execution_mode: Option<ExecutionMode>,
    pub media_source: Option<Box<dyn FrameSource>>,
    pub runtime: Option<RuntimeChoice>,
    pub model: Option<ModelChoice>,
    pub tracker_config: Option<Box<dyn Tracker>>,
    pub subscribers: Vec<Box<dyn TrackingSubscriber>>,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            execution_mode: None,
            media_source: None,
            runtime: None,
            model: None,
            tracker_config: None,
            subscribers: vec![],
        }
    }

    pub fn set_execution_mode (&mut self, execution_mode: ExecutionMode) -> &mut PipelineBuilder {
        self.execution_mode = Some(execution_mode);
        self
    }

    pub fn set_media_source (&mut self, media_source_choice: MediaSourceChoice) -> Result<&mut Self, AforaError> {
        let src = MediaSourceFactory::build(media_source_choice)?;
        self.media_source = Some(src);
        Ok(self)
    }

    pub fn set_runtime (&mut self, runtime_choice: RuntimeChoice) -> &mut Self {
        self.runtime = Some(runtime_choice);
        self
    }

    pub fn set_model (&mut self, model_choice: ModelChoice) -> &mut Self {
        self.model = Some(model_choice);
        self
    }

    pub fn set_tracker_config (&mut self, tracker_config_choice: TrackerChoice) -> Result<&mut Self, AforaError> {
        let tracker = TrackerFactory::build(tracker_config_choice)?;
        self.tracker_config = Some(tracker);
        Ok(self)
    }

    pub fn add_subscriber (&mut self, subscriber_choice: Box<dyn TrackingSubscriber>) -> &mut Self {
        self.subscribers.push(subscriber_choice);
        self
    }

    pub fn build(&mut self) -> Result<Box<dyn Pipeline>, AforaError> {
        let execution_mode = self.execution_mode.take().ok_or_else(|| {
            AforaError::ConfigurationError(
                "Execution mode not configured.".into(),
            )
        })?;

        let media_source = self.media_source.take().ok_or_else(|| {
            AforaError::ConfigurationError(
                "Media source not configured.".into(),
            )
        })?;

        let runtime = self.runtime.take().ok_or_else(|| {
            AforaError::ConfigurationError(
                "Runtime not configured.".into(),
            )
        })?;

        let model = self.model.take().ok_or_else(|| {
            AforaError::ConfigurationError(
                "Model not configured.".into(),
            )
        })?;

        let tracker = self.tracker_config.take().ok_or_else(|| {
            AforaError::ConfigurationError(
                "Tracker not configured.".into(),
            )
        })?;

        let detector = DetectorFactory::build(runtime, model)?;

        let config = PipelineConfig {
            execution_mode,
            media_source,
            detector,
            tracker,
            subscribers: std::mem::take(&mut self.subscribers),
        };

        PipelineFactory::build(config)
    }
}