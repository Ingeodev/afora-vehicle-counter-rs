use std::path::{ PathBuf};

use ab_glyph::FontArc;
use ffmpeg_next as ffmpeg;
use ffmpeg::codec;
use ffmpeg::encoder;
use ffmpeg::format::{self, Pixel};
use ffmpeg::software::scaling::{context::Context as ScalingContext, flag::Flags};
use ffmpeg::util::frame::video::Video as FfmpegVideoFrame;
use ffmpeg::{Dictionary, Rational};
use image::RgbImage;

use crate::core::afora_error::AforaError;
use crate::features::tracker::domain::tracking_output::TrackingOutput;
use crate::shared::domain::frame::Frame;
use crate::shared::domain::overlay::{draw_overlays, load_default_font};
use ffmpeg::encoder::video::Encoder as OpenedVideoEncoder;

pub struct VideoWriter {
    octx: format::context::Output,
    encoder: OpenedVideoEncoder,
    scaler: ScalingContext,
    font: FontArc,
    width: u32,
    height: u32,
    frame_index: i64,
    stream_index: usize,
    ost_time_base: Rational,
    finished: bool,
}

impl VideoWriter {
    /// `fps` se usa como time_base del stream (1/fps) y como frame_rate del encoder.
    /// `crf` controla calidad (menor = mejor calidad/más peso, 18-28 es rango típico).
    pub fn new(
        output_path: PathBuf,
        width: u32,
        height: u32,
        fps: u32,
        crf: u32,
    ) -> Result<Self, AforaError> {
        ffmpeg::init().map_err(|e| AforaError::MediaError(e.to_string()))?;

        let mut octx =
            format::output(&output_path).map_err(|e| AforaError::MediaError(e.to_string()))?;

        let codec = encoder::find(codec::Id::H264).ok_or_else(|| {
            AforaError::MediaError(
                "Encoder H264 (libx264) no disponible en esta build de FFmpeg".into(),
            )
        })?;

        let global_header = octx.format().flags().contains(format::Flags::GLOBAL_HEADER);

        let mut encoder_ctx = codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()
            .map_err(|e| AforaError::MediaError(e.to_string()))?;

        let time_base = Rational(1, fps as i32);
        encoder_ctx.set_width(width);
        encoder_ctx.set_height(height);
        encoder_ctx.set_format(Pixel::YUV420P);
        encoder_ctx.set_time_base(time_base);
        encoder_ctx.set_frame_rate(Some(Rational(fps as i32, 1)));

        if global_header {
            encoder_ctx.set_flags(codec::Flags::GLOBAL_HEADER);
        }

        let mut x264_opts = Dictionary::new();
        x264_opts.set("preset", "medium");
        x264_opts.set("crf", &crf.to_string());

        // `open_with` consume el `video::Video` sin abrir y devuelve el encoder
        // YA abierto, cuyo tipo real es `ffmpeg::encoder::Video` (alias top-level).
        let opened_encoder: OpenedVideoEncoder = encoder_ctx
            .open_with(x264_opts)
            .map_err(|e| AforaError::MediaError(e.to_string()))?;

        let stream_index;
        {
            // Bloque propio para que el borrow mutable de `octx` a través de `ost`
            // termine aquí explícitamente, antes de llamar a octx.write_header().
            let mut ost = octx
                .add_stream(codec)
                .map_err(|e| AforaError::MediaError(e.to_string()))?;
            stream_index = ost.index();
            ost.set_parameters(&opened_encoder);
        }

        octx.write_header()
            .map_err(|e| AforaError::MediaError(e.to_string()))?;

        let ost_time_base = octx.stream(stream_index).unwrap().time_base();

        let scaler = ScalingContext::get(
            Pixel::RGB24,
            width,
            height,
            Pixel::YUV420P,
            width,
            height,
            Flags::BILINEAR,
        )
            .map_err(|e| AforaError::MediaError(e.to_string()))?;

        Ok(Self {
            octx,
            encoder: opened_encoder,
            scaler,
            font: load_default_font()?,
            width,
            height,
            frame_index: 0,
            stream_index,
            ost_time_base,
            finished: false,
        })
    }

    /// Dibuja los tracks sobre el frame y lo entrega al encoder. Se puede llamar
    /// una vez por cada Frame que produzca tu FrameSource.
    pub fn write(&mut self, frame: &Frame, tracks: &[TrackingOutput]) -> Result<(), AforaError> {
        if frame.width != self.width || frame.height != self.height {
            return Err(AforaError::PostprocessError(format!(
                "Frame {}x{} no coincide con el video {}x{}",
                frame.width, frame.height, self.width, self.height
            )));
        }

        let mut image = RgbImage::from_raw(frame.width, frame.height, frame.data.clone())
            .ok_or_else(|| AforaError::PostprocessError("Invalid RGB image.".into()))?;

        draw_overlays(&mut image, tracks, &self.font);

        let rgb_frame = rgb_image_to_ffmpeg_frame(&image);

        let mut yuv_frame = FfmpegVideoFrame::empty();
        self.scaler
            .run(&rgb_frame, &mut yuv_frame)
            .map_err(|e| AforaError::MediaError(e.to_string()))?;

        yuv_frame.set_pts(Some(self.frame_index));
        self.frame_index += 1;

        self.encoder
            .send_frame(&yuv_frame)
            .map_err(|e| AforaError::MediaError(e.to_string()))?;

        self.drain_encoder()
    }

    /// Hay que llamarlo al terminar de escribir todos los frames: hace flush
    /// del encoder y escribe el trailer del mp4. Sin esto el archivo queda corrupto.
    pub fn finish(&mut self) -> Result<(), AforaError> {
        self.encoder
            .send_eof()
            .map_err(|e| AforaError::MediaError(e.to_string()))?;
        self.drain_encoder()?;

        self.octx
            .write_trailer()
            .map_err(|e| AforaError::MediaError(e.to_string()))?;

        self.finished = true;
        Ok(())
    }

    fn drain_encoder(&mut self) -> Result<(), AforaError> {
        let mut packet = ffmpeg::Packet::empty();
        while self.encoder.receive_packet(&mut packet).is_ok() {
            packet.set_stream(self.stream_index);
            packet.rescale_ts(self.encoder.time_base(), self.ost_time_base);
            packet
                .write_interleaved(&mut self.octx)
                .map_err(|e| AforaError::MediaError(e.to_string()))?;
        }
        Ok(())
    }
}

impl Drop for VideoWriter {
    fn drop(&mut self) {
        // Red de seguridad: si alguien olvida llamar finish(), al menos evitamos
        // un mp4 sin trailer. Los errores aquí se ignoran porque Drop no puede fallar,
        // pero es preferible llamar finish() explícitamente para manejar el error.
        if !self.finished {
            let _ = self.encoder.send_eof();
            let mut packet = ffmpeg::Packet::empty();
            while self.encoder.receive_packet(&mut packet).is_ok() {
                packet.set_stream(self.stream_index);
                packet.rescale_ts(self.encoder.time_base(), self.ost_time_base);
                let _ = packet.write_interleaved(&mut self.octx);
            }
            let _ = self.octx.write_trailer();
        }
    }
}

/// Copia el RgbImage (packed, sin padding) al frame de ffmpeg respetando
/// el linesize que ffmpeg le asigne internamente al plane RGB24.
fn rgb_image_to_ffmpeg_frame(image: &RgbImage) -> FfmpegVideoFrame {
    let width = image.width();
    let height = image.height();
    let mut frame = FfmpegVideoFrame::new(Pixel::RGB24, width, height);

    let stride = frame.stride(0);
    let row_bytes = width as usize * 3;
    let src = image.as_raw();

    let dst = frame.data_mut(0);
    for y in 0..height as usize {
        let src_start = y * row_bytes;
        let dst_start = y * stride;
        dst[dst_start..dst_start + row_bytes]
            .copy_from_slice(&src[src_start..src_start + row_bytes]);
    }

    frame
}