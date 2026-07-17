use crate::core::afora_error::AforaError;
use crate::features::detector::application::helpers::{bytes_to_f32, non_max_suppression};
use crate::features::detector::domain::detection::{BoundingBox, Detection};
use crate::features::detector::ports::postprocessor::{Postprocessor, PostprocessorConfig};
use crate::features::detector::ports::tensor_output::TensorOutput;
use crate::shared::utilities::letterbox_transform::LetterboxTransform;

pub struct YolosPostprocessor {
    input_side: u32,
    conf_threshold: f32,
    nms_iou_threshold: f32,
    batch_size: u32,
}

impl YolosPostprocessor {
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

impl Postprocessor for YolosPostprocessor {

    fn create(cfg: PostprocessorConfig) -> Self {
        Self {
            input_side: cfg.input_side,
            conf_threshold: cfg.conf_threshold,
            nms_iou_threshold: cfg.nms_iou_threshold,
            batch_size: cfg.batch_size,
        }
    }
    fn postprocess(&self, output: TensorOutput, original_size: (u32, u32)) -> Result<Vec<Vec<Detection>>, AforaError> {
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

    fn name() -> &'static str {
        "Yolo-11-S Postprocessor"
    }
}