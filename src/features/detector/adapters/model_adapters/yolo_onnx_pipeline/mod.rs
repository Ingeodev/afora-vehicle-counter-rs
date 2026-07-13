// =============================================================================
// YOLO ONNX OPTIMIZED PIPELINE MODULE
// =============================================================================
//
// Pipeline de alto rendimiento para modelos YOLO en ONNX.
//
// Estructura:
// - pipeline.rs: YoloOnnxOptimizedPipeline (implementa ModelPipeline)
// - preprocessing/: Motor de preprocessing optimizado
// - postprocessing.rs: Decodificación de salida YOLO

pub mod pipeline;
pub mod postprocessing;
pub mod preprocessing;

pub use pipeline::YoloOnnxOptimizedPipeline;
