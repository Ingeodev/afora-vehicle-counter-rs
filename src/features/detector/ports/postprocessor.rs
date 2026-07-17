use crate::core::afora_error::AforaError;
use crate::features::detector::domain::detection::Detection;
use crate::features::detector::ports::tensor_output::TensorOutput;

pub trait Postprocessor {

    fn create(cfg: PostprocessorConfig) -> Self;
    fn postprocess(
        &self,
        output: TensorOutput,
        original_size: (u32, u32),
    ) -> Result<Vec<Vec<Detection>>, AforaError>;

    fn name() -> &'static str;

}

pub struct PostprocessorConfig {
    pub input_side: u32,
    pub batch_size: u32,

    #[cfg(feature = "yolo11")]
    pub conf_threshold: f32,
    #[cfg(feature = "yolo11")]
    pub nms_iou_threshold: f32,
    
}