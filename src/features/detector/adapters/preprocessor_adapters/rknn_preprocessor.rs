use std::sync::Arc;
use crate::core::afora_error::AforaError;
use super::cpu::CPUPreprocessor;
use crate::features::detector::domain::preprocess_fallback::PreprocessFallbackPolicy;
use crate::features::detector::ports::preprocessor::{Preprocessor, PreprocessorConfig};
use crate::features::detector::ports::tensor_base::TensorSpec;
use crate::features::detector::ports::tensor_input::TensorInput;
use crate::shared::domain::frame::Frame;

pub struct RknnPreprocessor{
    fallback: PreprocessFallbackPolicy,
    cpu_preprocessor: CPUPreprocessor,
    tensor_spec: Arc<TensorSpec>,
}

impl RknnPreprocessor {

    fn preprocess_with_rknn(&self, frame: Vec<Arc<Frame>>) -> Result<TensorInput, AforaError> {
        let _ = frame;
        Err(AforaError::PreprocessError("RKNN Preprocessor not yet implemented".to_string()))
    }

    fn preprocess_with_cpu(&self, frame: Vec<Arc<Frame>>) -> Result<TensorInput, AforaError> {
        self.cpu_preprocessor.preprocess(frame)
    }
}

impl Preprocessor for RknnPreprocessor {
    fn preprocess(&self, frame: Vec<Arc<Frame>>) -> Result<TensorInput, AforaError> {
        let result = self.preprocess_with_rknn(frame.clone());
        if result.is_ok() {
            return result;
        }

        match &self.fallback {
            PreprocessFallbackPolicy::Cpu => self.preprocess_with_cpu(frame),
            PreprocessFallbackPolicy::Error(msg) => Err(AforaError::PreprocessError(msg.clone())),
        }
    }

    fn name() -> &'static str {
        "rk3588-preprocessor"
    }

    fn batch_size(&self) -> i64 {
        self.tensor_spec.shape[0]
    }

    fn create(config: PreprocessorConfig) -> Self {
        let cpu_preprocessor = CPUPreprocessor::create(PreprocessorConfig {
            fallback: PreprocessFallbackPolicy::Error("Error de CPU al preprocesar el frame".to_string()),
            target_spec: config.target_spec.clone(),
        });
        Self {
            fallback: config.fallback,
            cpu_preprocessor,
            tensor_spec: config.target_spec,
        }
    }

}