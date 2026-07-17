
#[cfg(feature = "yolo11")]
pub mod yolos_postprocessor;

#[cfg(feature = "yolo11")]
pub use yolos_postprocessor::YolosPostprocessor as PostprocessorImpl;