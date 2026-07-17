use std::sync::Arc;
use crate::core::afora_error::AforaError;
use crate::features::detector::domain::preprocess_fallback::PreprocessFallbackPolicy;
use crate::features::detector::ports::tensor_base::{TensorDType, TensorLayout, TensorSpec};
use crate::features::detector::ports::tensor_input::TensorInput;
use crate::shared::domain::frame::Frame;

pub trait Preprocessor {
    fn preprocess(&self, frame: Vec<Arc<Frame>>) -> Result<TensorInput, AforaError>;
    
    fn name() -> &'static str;

    fn batch_size(&self) -> i64;
    
    fn create(config: PreprocessorConfig) -> Self;
    
}

pub struct PreprocessorConfig {
    pub(crate) fallback: PreprocessFallbackPolicy,
    pub(crate) target_spec: Arc<TensorSpec>
}

impl PreprocessorConfig {
    pub fn new(fallback: PreprocessFallbackPolicy) -> Self {
        Self {
            fallback,
            target_spec: Arc::new(TensorSpec::new(
                vec![],
                TensorDType::F32,
                TensorLayout::Nchw,
            )),
        }
    }
}