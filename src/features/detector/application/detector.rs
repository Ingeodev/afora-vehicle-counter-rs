use std::sync::Arc;
use crate::core::afora_error::AforaError;
use crate::features::detector::adapters::postprocessor_adapters::PostprocessorImpl;
use crate::features::detector::adapters::preprocessor_adapters::PreprocessorImpl;
use crate::features::detector::adapters::runtime_adapters::RuntimeImpl;
use crate::features::detector::domain::detection::Detection;
use crate::features::detector::ports::inference_runtime::{InferenceRuntime, InferenceRuntimeConfig};
use crate::features::detector::ports::postprocessor::{Postprocessor, PostprocessorConfig};
use crate::features::detector::ports::preprocessor::{Preprocessor, PreprocessorConfig};
use crate::features::detector::ports::tensor_base::TensorSpec;
use crate::shared::domain::frame::Frame;
use crate::stacktrace;

pub struct Detector {
    pub(crate) preprocessor: PreprocessorImpl,
    runtime: RuntimeImpl,
    postprocessor: PostprocessorImpl,
}

impl Detector {
    pub fn new(
        preprocessor_config: PreprocessorConfig,
        runtime_config: InferenceRuntimeConfig,
        postprocessor_config: PostprocessorConfig,
    ) -> Result<Detector, AforaError> {
        let runtime = RuntimeImpl::load(runtime_config)?;
        let runtime_spec = runtime.input_spec();
        
        

        if !runtime_spec.matches_logical_shape(
            postprocessor_config.batch_size,
            3,
            postprocessor_config.input_side,
            postprocessor_config.input_side,
        ) {
            return Err(AforaError::ShapeMismatch {
                expected: (
                    postprocessor_config.batch_size,
                    3,
                    postprocessor_config.input_side,
                    postprocessor_config.input_side,
                ),
                actual: runtime_spec.shape.clone(),
            });
        }
        
        let mut  preprocess_tensor_spec = runtime_spec.clone();
        
        preprocess_tensor_spec.shape[0] = postprocessor_config.batch_size as i64;
        preprocess_tensor_spec.shape[2] = postprocessor_config.input_side as i64;
        preprocess_tensor_spec.shape[3] = postprocessor_config.input_side as i64;

        let preprocessor = PreprocessorImpl::create(PreprocessorConfig {
            target_spec: Arc::new(preprocess_tensor_spec),
            fallback: preprocessor_config.fallback,
        });

        let postprocessor = PostprocessorImpl::create(postprocessor_config);

        Ok(Self {
            preprocessor,
            runtime,
            postprocessor,
        })
    }

    pub fn detect(&mut self, frames: Vec<Arc<Frame>>) -> Result<Vec<Vec<Detection>>, AforaError> {
        if frames.is_empty() {
            return Err(AforaError::PreprocessError(
                "El batch está vacío".into(),
            ));
        }

        let input = stacktrace!("detection_preprocessing", "detect",
            self.preprocessor.preprocess(frames.clone())
        )?;

        let output = stacktrace!("detection_inference", "detect",
            self.runtime.run(&input)
        )?;

        stacktrace!("detection_postprocessing", "detect",
            self.postprocessor.postprocess(output, frames[0].original_size())
        )
    }
}