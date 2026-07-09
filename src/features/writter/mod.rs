
use crate::core::afora_error::AforaError;
use crate::features::tracker::domain::tracking_output::TrackingOutput;
use crate::shared::domain::frame::Frame;
use std::path::Path;
use ab_glyph::{FontArc, PxScale};
use image::{Rgb, RgbImage};
use imageproc::{
    drawing::{draw_hollow_rect_mut, draw_text_mut},
    rect::Rect,
};


pub struct ImageWriter;
impl ImageWriter {
    pub fn write<P: AsRef<Path>>(
        &self,
        frame: &Frame,
        tracks: &[TrackingOutput],
        output_path: P,
    ) -> Result<(), AforaError> {

        let mut image = RgbImage::from_raw(
            frame.width,
            frame.height,
            frame.data.clone(),
        )
            .ok_or_else(|| {
                AforaError::PostprocessError(
                    "Invalid RGB image.".into(),
                )
            })?;

        // Puedes colocar cualquier fuente TTF dentro de assets/fonts/
        let font = FontArc::try_from_vec(
            std::fs::read("assets/fonts/Roboto/Roboto-VariableFont_wdth,wght.ttf")
                .map_err(|e| AforaError::PostprocessError(e.to_string()))?,
        )
            .map_err(|e| AforaError::PostprocessError(e.to_string()))?;

        let scale = PxScale::from(18.0);

        for track in tracks {

            let bbox = &track.bbox;

            let x = bbox.x1.max(0.0).round() as i32;
            let y = bbox.y1.max(0.0).round() as i32;

            let w = (bbox.x2 - bbox.x1)
                .max(1.0)
                .round() as u32;

            let h = (bbox.y2 - bbox.y1)
                .max(1.0)
                .round() as u32;

            draw_hollow_rect_mut(
                &mut image,
                Rect::at(x, y).of_size(w, h),
                Rgb([255, 0, 0]),
            );

            draw_text_mut(
                &mut image,
                Rgb([255, 255, 0]),
                x,
                (y - 20).max(0),
                scale,
                &font,
                &track.id.to_string(),
            );
        }

        image
            .save(output_path)
            .map_err(|e| AforaError::PostprocessError(e.to_string()))?;

        Ok(())
    }
}