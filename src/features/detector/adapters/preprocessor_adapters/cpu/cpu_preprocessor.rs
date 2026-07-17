use std::sync::Arc;
use crate::core::afora_error::AforaError;
use crate::features::detector::domain::preprocess_fallback::PreprocessFallbackPolicy;
use crate::features::detector::ports::preprocessor::{Preprocessor, PreprocessorConfig};
use crate::features::detector::ports::tensor_base::{TensorLayout, TensorSpec};
use crate::features::detector::ports::tensor_input::TensorInput;
use crate::shared::domain::frame::Frame;

use super::engine::PreprocessingEngine;

pub struct CPUPreprocessor {
    fallback: PreprocessFallbackPolicy,
    tensor_spec: Arc<TensorSpec>,
    engine: PreprocessingEngine,
}

impl CPUPreprocessor {
    fn target_side_from_spec(spec: &TensorSpec) -> u32 {
        match spec.layout {
            TensorLayout::Nchw => spec.shape[2] as u32,
            TensorLayout::Nhwc => spec.shape[1] as u32,
        }
    }
}

impl Preprocessor for CPUPreprocessor {
    fn preprocess(&self, frame: Vec<Arc<Frame>>) -> Result<TensorInput, AforaError> {
        self.engine.process_batch(&frame, &self.tensor_spec)
    }

    fn create(config: PreprocessorConfig) -> Self {
        let target_side = Self::target_side_from_spec(&config.target_spec);
        Self {
            fallback: config.fallback,
            tensor_spec: config.target_spec,
            engine: PreprocessingEngine::new(target_side),
        }
    }

    fn batch_size(&self) -> i64 {
        self.tensor_spec.shape[0]
    }

    fn name() -> &'static str {
        "CPU preprocessor"
    }
}
