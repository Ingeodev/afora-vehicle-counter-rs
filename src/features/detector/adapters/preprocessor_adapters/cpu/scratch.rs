use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{PixelType, ResizeAlg, ResizeOptions, Resizer};

use crate::core::afora_error::AforaError;
use crate::shared::domain::frame::Frame;
use crate::shared::utilities::letterbox_transform::LetterboxTransform;
use crate::stacktrace;

use super::lut::{NORM_F32_LUT, PAD_F32, PAD_I8, PAD_U8};

pub struct ScratchContext {
    resizer: Resizer,
    dst_image: Option<Image<'static>>,
    current_dst_size: (u32, u32),
    pub(crate) target_side: u32,
    resize_options: ResizeOptions,
}

impl ScratchContext {
    pub fn new(target_side: u32) -> Self {
        let resizer = Resizer::new();

        Self {
            resizer,
            dst_image: None,
            current_dst_size: (0, 0),
            target_side,
            resize_options: ResizeOptions::new()
                .resize_alg(ResizeAlg::Convolution(fast_image_resize::FilterType::Bilinear)),
        }
    }

    #[inline]
    fn ensure_dst_capacity(&mut self, new_width: u32, new_height: u32) {
        if self.current_dst_size != (new_width, new_height) {
            self.dst_image = Some(Image::new(new_width, new_height, PixelType::U8x3));
            self.current_dst_size = (new_width, new_height);
        }
    }

    fn resize_frame(&mut self, frame: &Frame, letterbox: &LetterboxTransform) -> Result<(), AforaError> {
        let src = stacktrace!("resize_create_imageref", "preprocessing", {
            ImageRef::new(frame.width, frame.height, &frame.data, PixelType::U8x3)
                .map_err(|e| AforaError::PreprocessError(format!("ImageRef creation failed: {e}")))
        })?;

        stacktrace!("resize_ensure_capacity", "preprocessing", {
            self.ensure_dst_capacity(letterbox.new_width, letterbox.new_height);
        });

        let dst = self.dst_image.as_mut().unwrap();
        stacktrace!("resize_simd", "preprocessing", {
            self.resizer
                .resize(&src, dst, &self.resize_options)
                .map_err(|e| AforaError::PreprocessError(format!("Resize failed: {e}")))
        })?;

        Ok(())
    }

    pub fn process_frame_nchw_f32(
        &mut self,
        frame: &Frame,
        letterbox: &LetterboxTransform,
        out: &mut [f32],
    ) -> Result<(), AforaError> {
        stacktrace!("preprocess_resize", "preprocessing", {
            self.resize_frame(frame, letterbox)?;
        });

        stacktrace!("preprocess_pack", "preprocessing", {
            self.pack_nchw_f32(letterbox, out);
        });

        Ok(())
    }

    pub fn process_frame_nhwc_u8(
        &mut self,
        frame: &Frame,
        letterbox: &LetterboxTransform,
        out: &mut [u8],
    ) -> Result<(), AforaError> {
        stacktrace!("preprocess_resize", "preprocessing", {
            self.resize_frame(frame, letterbox)?;
        });

        stacktrace!("preprocess_pack", "preprocessing", {
            self.pack_nhwc_u8(letterbox, out);
        });

        Ok(())
    }

    pub fn process_frame_nchw_u8(
        &mut self,
        frame: &Frame,
        letterbox: &LetterboxTransform,
        out: &mut [u8],
    ) -> Result<(), AforaError> {
        stacktrace!("preprocess_resize", "preprocessing", {
            self.resize_frame(frame, letterbox)?;
        });

        stacktrace!("preprocess_pack", "preprocessing", {
            self.pack_nchw_u8(letterbox, out);
        });

        Ok(())
    }

    pub fn process_frame_nchw_i8(
        &mut self,
        frame: &Frame,
        letterbox: &LetterboxTransform,
        out: &mut [i8],
    ) -> Result<(), AforaError> {
        stacktrace!("preprocess_resize", "preprocessing", {
            self.resize_frame(frame, letterbox)?;
        });

        stacktrace!("preprocess_pack", "preprocessing", {
            self.pack_nchw_i8(letterbox, out);
        });

        Ok(())
    }

    pub fn process_frame_nhwc_i8(
        &mut self,
        frame: &Frame,
        letterbox: &LetterboxTransform,
        out: &mut [i8],
    ) -> Result<(), AforaError> {
        stacktrace!("preprocess_resize", "preprocessing", {
            self.resize_frame(frame, letterbox)?;
        });

        stacktrace!("preprocess_pack", "preprocessing", {
            self.pack_nhwc_i8(letterbox, out);
        });

        Ok(())
    }

    #[inline]
    fn pack_nchw_f32(&self, letterbox: &LetterboxTransform, out: &mut [f32]) {
        let side = self.target_side as usize;
        let plane = side * side;
        let scaled = self.dst_image.as_ref().unwrap().buffer();

        let pad_x = letterbox.pad_x.round() as usize;
        let pad_y = letterbox.pad_y.round() as usize;
        let new_w = letterbox.new_width as usize;
        let new_h = letterbox.new_height as usize;

        out.fill(PAD_F32);

        for y in 0..new_h {
            let dst_row = (y + pad_y) * side + pad_x;
            let src_row = y * new_w * 3;

            for x in 0..new_w {
                let s = src_row + x * 3;
                let d = dst_row + x;

                out[d] = NORM_F32_LUT[scaled[s] as usize];
                out[plane + d] = NORM_F32_LUT[scaled[s + 1] as usize];
                out[2 * plane + d] = NORM_F32_LUT[scaled[s + 2] as usize];
            }
        }
    }

    #[inline]
    fn pack_nhwc_u8(&self, letterbox: &LetterboxTransform, out: &mut [u8]) {
        let side = self.target_side as usize;
        let row_stride = side * 3;
        let scaled = self.dst_image.as_ref().unwrap().buffer();

        let pad_x = letterbox.pad_x.round() as usize;
        let pad_y = letterbox.pad_y.round() as usize;
        let new_w = letterbox.new_width as usize;
        let new_h = letterbox.new_height as usize;

        out.fill(PAD_U8);

        for y in 0..new_h {
            let dst_row = (y + pad_y) * row_stride + pad_x * 3;
            let src_row = y * new_w * 3;

            for x in 0..new_w {
                let s = src_row + x * 3;
                let d = dst_row + x * 3;

                out[d] = scaled[s];
                out[d + 1] = scaled[s + 1];
                out[d + 2] = scaled[s + 2];
            }
        }
    }

    #[inline]
    fn pack_nchw_u8(&self, letterbox: &LetterboxTransform, out: &mut [u8]) {
        let side = self.target_side as usize;
        let plane = side * side;
        let scaled = self.dst_image.as_ref().unwrap().buffer();

        let pad_x = letterbox.pad_x.round() as usize;
        let pad_y = letterbox.pad_y.round() as usize;
        let new_w = letterbox.new_width as usize;
        let new_h = letterbox.new_height as usize;

        out.fill(PAD_U8);

        for y in 0..new_h {
            let dst_row = (y + pad_y) * side + pad_x;
            let src_row = y * new_w * 3;

            for x in 0..new_w {
                let s = src_row + x * 3;
                let d = dst_row + x;

                out[d] = scaled[s];
                out[plane + d] = scaled[s + 1];
                out[2 * plane + d] = scaled[s + 2];
            }
        }
    }

    #[inline]
    fn pack_nchw_i8(&self, letterbox: &LetterboxTransform, out: &mut [i8]) {
        let side = self.target_side as usize;
        let plane = side * side;
        let scaled = self.dst_image.as_ref().unwrap().buffer();

        let pad_x = letterbox.pad_x.round() as usize;
        let pad_y = letterbox.pad_y.round() as usize;
        let new_w = letterbox.new_width as usize;
        let new_h = letterbox.new_height as usize;

        out.fill(PAD_I8);

        for y in 0..new_h {
            let dst_row = (y + pad_y) * side + pad_x;
            let src_row = y * new_w * 3;

            for x in 0..new_w {
                let s = src_row + x * 3;
                let d = dst_row + x;

                out[d] = (scaled[s] as i16 - 128) as i8;
                out[plane + d] = (scaled[s + 1] as i16 - 128) as i8;
                out[2 * plane + d] = (scaled[s + 2] as i16 - 128) as i8;
            }
        }
    }

    #[inline]
    fn pack_nhwc_i8(&self, letterbox: &LetterboxTransform, out: &mut [i8]) {
        let side = self.target_side as usize;
        let row_stride = side * 3;
        let scaled = self.dst_image.as_ref().unwrap().buffer();

        let pad_x = letterbox.pad_x.round() as usize;
        let pad_y = letterbox.pad_y.round() as usize;
        let new_w = letterbox.new_width as usize;
        let new_h = letterbox.new_height as usize;

        out.fill(PAD_I8);

        for y in 0..new_h {
            let dst_row = (y + pad_y) * row_stride + pad_x * 3;
            let src_row = y * new_w * 3;

            for x in 0..new_w {
                let s = src_row + x * 3;
                let d = dst_row + x * 3;

                out[d] = (scaled[s] as i16 - 128) as i8;
                out[d + 1] = (scaled[s + 1] as i16 - 128) as i8;
                out[d + 2] = (scaled[s + 2] as i16 - 128) as i8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32, color: (u8, u8, u8)) -> Frame {
        let (r, g, b) = color;
        let data: Vec<u8> = (0..(width * height))
            .flat_map(|_| [r, g, b])
            .collect();

        Frame {
            data,
            width,
            height,
        }
    }

    #[test]
    fn scratch_context_reuses_resizer() {
        let mut ctx = ScratchContext::new(640);

        let frame1 = create_test_frame(1920, 1080, (255, 0, 0));
        let letterbox1 = LetterboxTransform::new((1920, 1080), 640);

        let mut buffer1 = vec![0.0f32; 3 * 640 * 640];
        ctx.process_frame_nchw_f32(&frame1, &letterbox1, &mut buffer1).unwrap();

        let frame2 = create_test_frame(1920, 1080, (0, 255, 0));
        let letterbox2 = LetterboxTransform::new((1920, 1080), 640);

        let mut buffer2 = vec![0.0f32; 3 * 640 * 640];
        ctx.process_frame_nchw_f32(&frame2, &letterbox2, &mut buffer2).unwrap();

        assert_eq!(ctx.current_dst_size, (letterbox2.new_width, letterbox2.new_height));
    }

    #[test]
    fn scratch_context_reallocates_on_size_change() {
        let mut ctx = ScratchContext::new(640);

        let frame1 = create_test_frame(1920, 1080, (255, 0, 0));
        let letterbox1 = LetterboxTransform::new((1920, 1080), 640);
        let size1 = (letterbox1.new_width, letterbox1.new_height);

        let mut buffer1 = vec![0.0f32; 3 * 640 * 640];
        ctx.process_frame_nchw_f32(&frame1, &letterbox1, &mut buffer1).unwrap();

        assert_eq!(ctx.current_dst_size, size1);

        let frame2 = create_test_frame(640, 480, (0, 255, 0));
        let letterbox2 = LetterboxTransform::new((640, 480), 640);
        let size2 = (letterbox2.new_width, letterbox2.new_height);

        let mut buffer2 = vec![0.0f32; 3 * 640 * 640];
        ctx.process_frame_nchw_f32(&frame2, &letterbox2, &mut buffer2).unwrap();

        assert_eq!(ctx.current_dst_size, size2);
        assert_ne!(size1, size2);
    }
}
