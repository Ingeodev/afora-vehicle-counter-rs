// =============================================================================
// YOLO ONNX OPTIMIZED PIPELINE
// =============================================================================
//
// Pipeline optimizado para modelos YOLOv8/YOLO11 exportados a ONNX.
// Usa el motor de preprocessing de alto rendimiento que:
// - Elimina copias innecesarias (ImageRef)
// - Reutiliza buffers (thread_local ScratchContext)
// - Escribe directamente al formato final (TensorWriter monomorfizado)
// - Usa LUT para normalización (sin divisiones en el hot path)

use std::sync::Arc;

use crate::core::afora_error::AforaError;
use crate::features::detector::domain::detection::Detection;
use crate::features::detector::ports::model_pipeline::ModelPipeline;
use crate::features::detector::ports::tensor_base::TensorSpec;
use crate::features::detector::ports::tensor_input::TensorInput;
use crate::features::detector::ports::tensor_output::TensorOutput;
use crate::shared::domain::frame::Frame;

use super::postprocessing::decode_yolo_output;
use super::preprocessing::PreprocessingEngine;

/// Pipeline optimizado para modelos YOLO en ONNX.
///
/// Diferencias con `YoloOnnxPipeline` original:
/// - Preprocessing usa `ImageRef` (zero-copy del frame)
/// - Thread-local scratch buffers (sin allocations por frame)
/// - TensorWriter monomorfizado (escritura directa al formato del TensorSpec)
/// - LUT de normalización (sin divisiones en el bucle interno)
///
/// La API pública es idéntica — solo cambia el rendimiento interno.
pub struct YoloOnnxOptimizedPipeline {
    /// Tamaño del lado del tensor cuadrado (ej. 640)
    input_side: u32,

    /// Umbral mínimo de confianza para detecciones
    conf_threshold: f32,

    /// Umbral de IoU para Non-Maximum Suppression
    nms_iou_threshold: f32,

    /// Tamaño del batch esperado
    batch_size: u32,

    /// Motor de preprocessing optimizado
    preprocessing_engine: PreprocessingEngine,
}

impl YoloOnnxOptimizedPipeline {
    /// Crea un nuevo pipeline optimizado.
    ///
    /// # Argumentos
    /// - `input_side`: Tamaño del lado del tensor (ej. 640 para YOLO 640x640)
    /// - `conf_threshold`: Umbral de confianza [0.0, 1.0]
    /// - `nms_iou_threshold`: Umbral de IoU para NMS [0.0, 1.0]
    /// - `batch_size`: Tamaño del batch esperado
    pub fn new(
        input_side: u32,
        conf_threshold: f32,
        nms_iou_threshold: f32,
        batch_size: u32,
    ) -> Self {
        Self {
            input_side,
            conf_threshold,
            nms_iou_threshold,
            batch_size,
            preprocessing_engine: PreprocessingEngine::new(input_side),
        }
    }
}

impl ModelPipeline for YoloOnnxOptimizedPipeline {
    fn preprocess(
        &self,
        frames: Vec<Arc<Frame>>,
        target_spec: &TensorSpec,
    ) -> Result<TensorInput, AforaError> {
        self.preprocessing_engine.process_batch(&frames, target_spec)
    }

    fn postprocess(
        &self,
        output: TensorOutput,
        original_size: (u32, u32),
    ) -> Result<Vec<Vec<Detection>>, AforaError> {
        decode_yolo_output(
            output,
            original_size,
            self.input_side,
            self.conf_threshold,
            self.nms_iou_threshold,
        )
    }

    fn model_name(&self) -> &'static str {
        "yolo_onnx_optimized"
    }

    fn expected_input_shape(&self) -> (u32, u32, u32, u32) {
        (self.batch_size, 3, self.input_side, self.input_side)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::detector::ports::tensor_base::{TensorDType, TensorLayout};

    fn create_test_frame(width: u32, height: u32) -> Arc<Frame> {
        let data = vec![128u8; (width * height * 3) as usize];
        Arc::new(Frame {
            data,
            width,
            height,
        })
    }

    #[test]
    fn pipeline_preprocesses_single_frame() {
        let pipeline = YoloOnnxOptimizedPipeline::new(640, 0.5, 0.45, 1);
        let frame = create_test_frame(1920, 1080);

        let spec = TensorSpec::new(
            vec![1, 3, 640, 640],
            TensorDType::F32,
            TensorLayout::Nchw,
        );

        let result = pipeline.preprocess(vec![frame], &spec).unwrap();

        assert_eq!(result.spec.shape[0], 1);
        assert_eq!(result.data.len(), 3 * 640 * 640 * 4);
    }

    #[test]
    fn pipeline_preprocesses_batch() {
        let pipeline = YoloOnnxOptimizedPipeline::new(640, 0.5, 0.45, 4);
        let frames: Vec<Arc<Frame>> = (0..4).map(|_| create_test_frame(1280, 720)).collect();

        let spec = TensorSpec::new(
            vec![4, 3, 640, 640],
            TensorDType::F32,
            TensorLayout::Nchw,
        );

        let result = pipeline.preprocess(frames, &spec).unwrap();

        assert_eq!(result.spec.shape[0], 4);
        assert_eq!(result.data.len(), 4 * 3 * 640 * 640 * 4);
    }

    #[test]
    fn pipeline_handles_u8_nhwc_spec() {
        let pipeline = YoloOnnxOptimizedPipeline::new(320, 0.5, 0.45, 1);
        let frame = create_test_frame(640, 480);

        let spec = TensorSpec::new(
            vec![1, 320, 320, 3],
            TensorDType::U8,
            TensorLayout::Nhwc,
        );

        let result = pipeline.preprocess(vec![frame], &spec).unwrap();

        assert_eq!(result.spec.shape[0], 1);
        assert_eq!(result.data.len(), 320 * 320 * 3);
    }

    #[test]
    fn pipeline_returns_correct_metadata() {
        let pipeline = YoloOnnxOptimizedPipeline::new(640, 0.5, 0.45, 2);

        assert_eq!(pipeline.model_name(), "yolo_onnx_optimized");
        assert_eq!(pipeline.expected_input_shape(), (2, 3, 640, 640));
    }
}
