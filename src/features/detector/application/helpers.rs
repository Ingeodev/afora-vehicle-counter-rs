use image::RgbImage;
use crate::features::detector::domain::detection::{BoundingBox, Detection};

pub fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// HWC (RGB, u8) -> CHW (f32, normalizado [0,1]) — layout esperado por
/// modelos exportados con la convención estándar de PyTorch/ONNX (NCHW).
pub fn hwc_u8_to_chw_f32_normalized(img: &RgbImage) -> Vec<f32> {
    let (w, h) = img.dimensions();
    let plane_size = (w * h) as usize;
    let mut out = vec![0f32; 3 * plane_size];

    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x, y);
            let idx = (y * w + x) as usize;
            out[idx] = px[0] as f32 / 255.0; // R
            out[plane_size + idx] = px[1] as f32 / 255.0; // G
            out[2 * plane_size + idx] = px[2] as f32 / 255.0; // B
        }
    }
    out
}

pub fn iou(a: &BoundingBox, b: &BoundingBox) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);

    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let area_b = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
    let union = area_a + area_b - inter;

    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// NMS por clase: ordena por confianza descendente y descarta cajas de la
/// MISMA clase que se solapen por encima del umbral con una ya aceptada.
pub fn non_max_suppression(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<Detection> = Vec::new();
    for det in detections {
        let suppressed = kept
            .iter()
            .any(|k| k.class_id == det.class_id && iou(&k.bbox, &det.bbox) > iou_threshold);
        if !suppressed {
            kept.push(det);
        }
    }
    kept
}