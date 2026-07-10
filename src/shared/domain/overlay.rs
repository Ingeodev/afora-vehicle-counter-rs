// shared/domain/overlay.rs
use ab_glyph::{FontArc, PxScale};
use image::{Rgb, RgbImage};
use imageproc::{
    drawing::{draw_hollow_rect_mut, draw_text_mut},
    rect::Rect,
};

use crate::features::tracker::domain::tracking_output::TrackingOutput;

/// Dibuja bboxes + ids sobre la imagen. Compartido entre ImageWriter y VideoWriter
/// para no duplicar la lógica de overlay ni recargar la fuente en cada frame.
pub fn draw_overlays(image: &mut RgbImage, tracks: &[TrackingOutput], font: &FontArc) {
    let scale = PxScale::from(18.0);
    for track in tracks {
        let bbox = &track.bbox;
        let x = bbox.x1.max(0.0).round() as i32;
        let y = bbox.y1.max(0.0).round() as i32;
        let w = (bbox.x2 - bbox.x1).max(1.0).round() as u32;
        let h = (bbox.y2 - bbox.y1).max(1.0).round() as u32;

        draw_hollow_rect_mut(image, Rect::at(x, y).of_size(w, h), Rgb([255, 0, 0]));
        draw_text_mut(
            image,
            Rgb([255, 255, 0]),
            x,
            (y - 20).max(0),
            scale,
            font,
            &track.id.to_string(),
        );
    }
}

pub fn load_default_font() -> Result<FontArc, crate::core::afora_error::AforaError> {
    FontArc::try_from_vec(
        std::fs::read("assets/fonts/Roboto/Roboto-VariableFont_wdth,wght.ttf")
            .map_err(|e| crate::core::afora_error::AforaError::PostprocessError(e.to_string()))?,
    )
        .map_err(|e| crate::core::afora_error::AforaError::PostprocessError(e.to_string()))
}