use std::sync::Arc;
use crate::core::afora_error::AforaError;
use super::cpu::CPUPreprocessor;
use crate::features::detector::domain::preprocess_fallback::PreprocessFallbackPolicy;
use crate::features::detector::ports::preprocessor::{Preprocessor, PreprocessorConfig};
use crate::features::detector::ports::tensor_base::TensorSpec;
use crate::features::detector::ports::tensor_input::TensorInput;
use crate::shared::domain::frame::Frame;

pub struct CudaPreprocessor{
    fallback: PreprocessFallbackPolicy,
    cpu_preprocessor: CPUPreprocessor,
    tensor_spec: Arc<TensorSpec>,
}

impl CudaPreprocessor {
    fn preprocess_with_cuda(&self, frame: Vec<Arc<Frame>>) -> Result<TensorInput, AforaError> {
        Err(AforaError::PreprocessError("CUDA Preprocessors are not supported".to_string()))
    }

    fn preprocess_with_cpu(&self, frame: Vec<Arc<Frame>>) -> Result<TensorInput, AforaError> {
        self.cpu_preprocessor.preprocess(frame.clone())
    }
}

impl Preprocessor for CudaPreprocessor {
    fn preprocess(&self, frame: Vec<Arc<Frame>>) -> Result<TensorInput, AforaError> {
        let result = self.preprocess_with_cuda(frame.clone());
        if result.is_ok() {
            return result;
        }

        match &self.fallback {
            PreprocessFallbackPolicy::Cpu => self.preprocess_with_cpu(frame),
            PreprocessFallbackPolicy::Error(msg) => Err(AforaError::PreprocessError(msg.clone())),
        }
    }

    fn name() -> &'static str {
        "cuda-preprocessor"
    }
    
    fn batch_size(&self) -> i64 {
        self.tensor_spec.shape[0]
    }

    fn create(config: PreprocessorConfig) -> Self {
        Self {
            fallback: config.fallback,
            cpu_preprocessor: CPUPreprocessor::create(PreprocessorConfig {
                fallback: PreprocessFallbackPolicy::Error("Error de CPU al preprocesar el frame".to_string()),
                target_spec: config.target_spec.clone()
            }),
            tensor_spec: config.target_spec
        }
    }

}