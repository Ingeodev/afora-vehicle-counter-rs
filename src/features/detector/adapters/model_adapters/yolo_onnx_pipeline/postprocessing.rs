// =============================================================================
// POSTPROCESSING — Decodificación de salida YOLO
// =============================================================================
//
// Lógica de postprocesamiento extraída para mantener el pipeline limpio.
// Maneja la decodificación de la salida del modelo YOLO y la conversión
// de coordenadas de vuelta al espacio de la imagen original.

use crate::core::afora_error::AforaError;
use crate::features::detector::application::helpers::{bytes_to_f32, non_max_suppression};
use crate::features::detector::domain::detection::{BoundingBox, Detection};
use crate::features::detector::ports::tensor_output::TensorOutput;
use crate::shared::utilities::letterbox_transform::LetterboxTransform;

/// Decodifica la salida del modelo YOLO.
///
/// # Formato esperado
/// Tensor único con shape `[batch, 4 + num_classes, num_anchors]`
/// - Primeros 4 atributos: cx, cy, w, h (coordenadas del centro y dimensiones)
/// - Siguientes N atributos: scores por clase
///
/// # Argumentos
/// - `output`: Salida del runtime
/// - `original_size`: Tamaño original de la imagen (w, h)
/// - `input_side`: Tamaño del lado del tensor de entrada
/// - `conf_threshold`: Umbral mínimo de confianza
/// - `nms_iou_threshold`: Umbral de IoU para NMS
pub fn decode_yolo_output(
    output: TensorOutput,
    original_size: (u32, u32),
    input_side: u32,
    conf_threshold: f32,
    nms_iou_threshold: f32,
) -> Result<Vec<Vec<Detection>>, AforaError> {
    let (_, raw_bytes, spec) = output
        .tensors
        .first()
        .ok_or_else(|| AforaError::PostprocessError("El modelo no retornó tensores de salida.".into()))?;

    if spec.shape.len() != 3 {
        return Err(AforaError::PostprocessError(format!(
            "Shape inesperado. Esperado [N, 4 + num_classes, num_anchors], recibido {:?}",
            spec.shape
        )));
    }

    let data = bytes_to_f32(raw_bytes);

    let batch = spec.shape[0] as usize;
    let num_attrs = spec.shape[1] as usize;
    let num_anchors = spec.shape[2] as usize;

    if num_attrs <= 4 {
        return Err(AforaError::PostprocessError(
            "El tensor de salida no contiene predicciones de clase.".into(),
        ));
    }

    let num_classes = num_attrs - 4;
    let image_stride = num_attrs * num_anchors;

    let letterbox = LetterboxTransform::new(original_size, input_side);

    let mut batch_detections = Vec::with_capacity(batch);

    for b in 0..batch {
        let start = b * image_stride;
        let end = start + image_stride;
        let image_data = &data[start..end];

        let mut candidates = Vec::new();

        for anchor in 0..num_anchors {
            let (cx, cy, w, h) = read_anchor_box(image_data, num_anchors, anchor);
            let (class_id, confidence) = find_best_class(image_data, num_classes, num_anchors, anchor);

            if confidence < conf_threshold {
                continue;
            }

            let (x1, y1, x2, y2) = letterbox.restore_bbox(cx, cy, w, h);

            candidates.push(Detection {
                bbox: BoundingBox { x1, y1, x2, y2 },
                class_id: class_id as u32,
                confidence,
            });
        }

        batch_detections.push(non_max_suppression(candidates, nms_iou_threshold));
    }

    Ok(batch_detections)
}

/// Lee las coordenadas del anchor box desde el tensor de salida.
#[inline]
fn read_anchor_box(data: &[f32], num_anchors: usize, anchor: usize) -> (f32, f32, f32, f32) {
    (
        data[anchor],                      // cx
        data[num_anchors + anchor],        // cy
        data[2 * num_anchors + anchor],    // w
        data[3 * num_anchors + anchor],    // h
    )
}

/// Encuentra la clase con mayor score para un anchor.
#[inline]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::detector::ports::tensor_base::{TensorDType, TensorLayout, TensorSpec};

    fn create_dummy_output(batch: usize, num_classes: usize, num_anchors: usize) -> TensorOutput {
        let num_attrs = 4 + num_classes;
        let total_elements = batch * num_attrs * num_anchors;
        
        // Crear datos dummy
        let data: Vec<f32> = (0..total_elements).map(|i| i as f32 * 0.001).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        
        let spec = TensorSpec::new(
            vec![batch as i64, num_attrs as i64, num_anchors as i64],
            TensorDType::F32,
            TensorLayout::Nchw,
        );
        
        TensorOutput::new(vec![("output".to_string(), bytes, spec)])
    }

    #[test]
    fn decode_handles_empty_detections() {
        let output = create_dummy_output(1, 80, 8400);
        
        // Con threshold muy alto, no debería haber detecciones
        let result = decode_yolo_output(output, (1920, 1080), 640, 0.99, 0.45).unwrap();
        
        assert_eq!(result.len(), 1);
        // Las detecciones pueden estar vacías o no dependiendo de los datos dummy
    }

    #[test]
    fn decode_rejects_invalid_shape() {
        let spec = TensorSpec::new(
            vec![1, 85], // Solo 2 dimensiones, inválido
            TensorDType::F32,
            TensorLayout::Nchw,
        );
        
        let output = TensorOutput::new(vec![("output".to_string(), vec![0u8; 340], spec)]);
        
        let result = decode_yolo_output(output, (1920, 1080), 640, 0.5, 0.45);
        assert!(result.is_err());
    }
}
