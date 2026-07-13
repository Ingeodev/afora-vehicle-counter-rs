// =============================================================================
// MODEL PIPELINE ADAPTER — YOLOv8 / YOLO11 (formato de salida Ultralytics)
// =============================================================================

use std::sync::Arc;
use rayon::prelude::*;

use crate::core::afora_error::AforaError;
use crate::features::detector::application::helpers::{
    bytes_to_f32, f32_slice_to_bytes, non_max_suppression,
};
use crate::features::detector::domain::detection::{BoundingBox, Detection};
use crate::features::detector::ports::model_pipeline::ModelPipeline;
use crate::features::detector::ports::tensor_base::{TensorDType, TensorSpec};
use crate::features::detector::ports::tensor_input::TensorInput;
use crate::features::detector::ports::tensor_output::TensorOutput;
use crate::shared::domain::frame::Frame;
use crate::shared::utilities::letterbox_transform::LetterboxTransform;

/// Pipeline para modelos YOLOv8/YOLO11 exportados a ONNX SIN NMS embebido.
/// Salida esperada: tensor único [1, 4 + num_classes, num_anchors].
///
/// Si tu export sí incluye NMS en el grafo, house esa variante en otro
/// struct (ej. `YoloOnnxNmsEmbeddedPipeline`) en vez de ramificar con un if aquí.
pub struct YoloOnnxPipeline {
    input_side: u32,
    conf_threshold: f32,
    nms_iou_threshold: f32,
    batch_size: u32,
}

impl YoloOnnxPipeline {
    pub fn new(input_side: u32, conf_threshold: f32, nms_iou_threshold: f32, batch_size: u32) -> Self {
        Self {
            input_side,
            conf_threshold,
            nms_iou_threshold,
            batch_size,
        }
    }

    fn build_tensor_input(
        tensor: Vec<f32>,
        target_spec: &TensorSpec,
    ) -> Result<TensorInput, AforaError> {
        match target_spec.dtype {
            TensorDType::F32 => Ok(TensorInput::new(
                f32_slice_to_bytes(&tensor),
                target_spec.clone(),
            )),
            TensorDType::U8 | TensorDType::I8 => Ok(TensorInput::new(
                Self::quantize_tensor(&tensor),
                target_spec.clone(),
            )),
        }
    }

    fn quantize_tensor(tensor: &[f32]) -> Vec<u8> {
        tensor
            .iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
            .collect()
    }

    fn read_anchor_box(data: &[f32], num_anchors: usize, anchor: usize) -> (f32, f32, f32, f32) {
        (
            data[anchor],
            data[num_anchors + anchor],
            data[2 * num_anchors + anchor],
            data[3 * num_anchors + anchor],
        )
    }

    fn find_best_class(
        data: &[f32],
        num_classes: usize,
        num_anchors: usize,
        anchor: usize,
    ) -> (usize, f32) {
        let mut best_class = 0;
        let mut best_score = f32::MIN;

        for class in 0..num_classes {
            let score = data[(4 + class) * num_anchors + anchor];
            if score > best_score {
                best_score = score;
                best_class = class;
            }
        }

        (best_class, best_score)
    }
}

impl ModelPipeline for YoloOnnxPipeline {
    fn preprocess(
        &self,
        frames: Vec<Arc<Frame>>,
        target_spec: &TensorSpec,
    ) -> Result<TensorInput, AforaError> {
        if frames.is_empty() {
            return Err(AforaError::PreprocessError("El batch está vacío".into()));
        }

        let side = self.input_side as usize;
        let elements_per_image = 3 * side * side;

        let mut batch = vec![0f32; elements_per_image * frames.len()];

        // Cada frame se resize+empaqueta de forma independiente; se reparte
        // entre los núcleos disponibles (A76/A55 en el Orange Pi) en vez de
        // procesar el batch en serie.
        batch
            .par_chunks_mut(elements_per_image)
            .zip(frames.par_iter())
            .try_for_each(|(chunk, frame)| -> Result<(), AforaError> {
                let letterbox = LetterboxTransform::new((frame.width, frame.height), self.input_side);
                letterbox.resize_and_pack_into(frame.width, frame.height, &frame.data, chunk)
            })?;

        let mut spec = target_spec.clone();
        spec.shape[0] = frames.len() as i64;
        spec.shape[2] = self.input_side as i64;
        spec.shape[3] = self.input_side as i64;

        Self::build_tensor_input(batch, &spec)
    }

    fn postprocess(
        &self,
        output: TensorOutput,
        original_size: (u32, u32),
    ) -> Result<Vec<Vec<Detection>>, AforaError> {
        let (_, raw_bytes, spec) = output
            .tensors
            .first()
            .ok_or_else(|| AforaError::PostprocessError("The model returned no output tensors.".into()))?;

        if spec.shape.len() != 3 {
            return Err(AforaError::PostprocessError(format!(
                "Unexpected output shape. Expected [N, 4 + num_classes, num_anchors], got {:?}",
                spec.shape
            )));
        }

        let data = bytes_to_f32(raw_bytes);

        let batch = spec.shape[0] as usize;
        let num_attrs = spec.shape[1] as usize;
        let num_anchors = spec.shape[2] as usize;

        if num_attrs <= 4 {
            return Err(AforaError::PostprocessError(
                "The output tensor contains no class predictions.".into(),
            ));
        }

        let num_classes = num_attrs - 4;
        let image_stride = num_attrs * num_anchors;

        let letterbox = LetterboxTransform::new(original_size, self.input_side);

        let mut batch_detections = Vec::with_capacity(batch);

        for b in 0..batch {
            let start = b * image_stride;
            let end = start + image_stride;
            let image_data = &data[start..end];

            let mut candidates = Vec::new();

            for anchor in 0..num_anchors {
                let (cx, cy, w, h) = Self::read_anchor_box(image_data, num_anchors, anchor);
                let (class_id, confidence) =
                    Self::find_best_class(image_data, num_classes, num_anchors, anchor);

                if confidence < self.conf_threshold {
                    continue;
                }

                let (x1, y1, x2, y2) = letterbox.restore_bbox(cx, cy, w, h);

                candidates.push(Detection {
                    bbox: BoundingBox { x1, y1, x2, y2 },
                    class_id: class_id as u32,
                    confidence,
                });
            }

            batch_detections.push(non_max_suppression(candidates, self.nms_iou_threshold));
        }

        Ok(batch_detections)
    }

    fn model_name(&self) -> &'static str {
        "yolo_onnx"
    }

    fn expected_input_shape(&self) -> (u32, u32, u32, u32) {
        (self.batch_size, 3, self.input_side, self.input_side)
    }
}