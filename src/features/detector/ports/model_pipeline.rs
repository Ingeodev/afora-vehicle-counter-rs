use std::sync::Arc;
use crate::core::afora_error::AforaError;
use crate::features::detector::domain::detection::Detection;
use crate::features::detector::ports::tensor_base::TensorSpec;
use crate::features::detector::ports::tensor_input::TensorInput;
use crate::features::detector::ports::tensor_output::TensorOutput;
use crate::shared::domain::frame::Frame;

/// Abstrae la lógica específica de una arquitectura de modelo
/// (PP-YOLOE+, YOLO11, RF-DETR...). NO toca hardware, NO sabe qué runtime
/// se usa — solo transforma Frame <-> TensorInput/Output <-> Detection.
pub trait ModelPipeline: Send + Sync {
    /// Convierte un frame de video al tensor que el modelo espera.
    /// Recibe el TensorSpec real del runtime destino para producir los bytes
    /// en el dtype/layout correctos (f32/NCHW para ONNX, u8/NHWC para RKNN, etc.)
    /// sin necesitar un `match` sobre qué runtime es.
    fn preprocess(&self, frame: Vec<Arc<Frame>>, target_spec: &TensorSpec) -> Result<TensorInput, AforaError>;

    /// Decodifica la salida cruda del runtime a detecciones en coordenadas
    /// de la imagen original (deshace resize/letterbox aplicado en preprocess).
    fn postprocess(
        &self,
        output: TensorOutput,
        original_size: (u32, u32),
    ) -> Result<Vec<Vec<Detection>>, AforaError>;

    fn model_name(&self) -> &'static str;

    /// Shape lógico (C, H, W) que la arquitectura fue entrenada a esperar,
    /// independiente del runtime — se valida contra el spec real al construir el Detector.
    fn expected_input_shape(&self) -> (u32, u32, u32, u32);

    //fn class_taxonomy(&self) -> &'static [&'static str];
}