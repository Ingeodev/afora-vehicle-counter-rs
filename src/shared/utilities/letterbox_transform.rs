use image::imageops::FilterType;
use image::{Rgb, RgbImage};

#[derive(Debug, Clone)]
pub struct LetterboxTransform {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    target_side: u32,
}

impl LetterboxTransform {
    pub fn new(original_size: (u32, u32), target_side: u32) -> Self {
        let (ow, oh) = (original_size.0 as f32, original_size.1 as f32);

        let scale = (target_side as f32 / ow)
            .min(target_side as f32 / oh);

        let new_width = ow * scale;
        let new_height = oh * scale;

        Self {
            scale,
            pad_x: (target_side as f32 - new_width) / 2.0,
            pad_y: (target_side as f32 - new_height) / 2.0,
            target_side,
        }
    }

    pub fn apply(&self, image: &RgbImage) -> RgbImage {
        let resized = image::imageops::resize(
            image,
            self.scaled_width(image.width()),
            self.scaled_height(image.height()),
            FilterType::Triangle,
        );

        let mut canvas = RgbImage::from_pixel(
            self.target_side,
            self.target_side,
            Rgb([114, 114, 114]),
        );

        image::imageops::overlay(
            &mut canvas,
            &resized,
            self.pad_x.round() as i64,
            self.pad_y.round() as i64,
        );

        canvas
    }

    pub fn restore_bbox(
        &self,
        cx: f32,
        cy: f32,
        w: f32,
        h: f32,
    ) -> (f32, f32, f32, f32) {
        (
            (cx - w * 0.5 - self.pad_x) / self.scale,
            (cy - h * 0.5 - self.pad_y) / self.scale,
            (cx + w * 0.5 - self.pad_x) / self.scale,
            (cy + h * 0.5 - self.pad_y) / self.scale,
        )
    }

    fn scaled_width(&self, width: u32) -> u32 {
        ((width as f32) * self.scale)
            .round()
            .max(1.0) as u32
    }

    fn scaled_height(&self, height: u32) -> u32 {
        ((height as f32) * self.scale)
            .round()
            .max(1.0) as u32
    }
}