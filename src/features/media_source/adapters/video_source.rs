use std::collections::VecDeque;
use std::path::PathBuf;

use ffmpeg_next as ffmpeg;
use ffmpeg::format::{input, Pixel};
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context as ScalingContext, flag::Flags};
use ffmpeg::util::frame::video::Video as FfmpegVideoFrame;

use crate::core::afora_error::AforaError;
use crate::shared::domain::frame::Frame;

pub struct VideoSource {
    ictx: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Video,
    scaler: ScalingContext,
    video_stream_index: usize,
    frame_queue: VecDeque<Frame>,
    eof_sent: bool,
    finished: bool,
}

impl VideoSource {
    pub fn new(path: impl Into<PathBuf>) -> Result<Self, AforaError> {
        let path = path.into();

        ffmpeg::init().map_err(|e| AforaError::MediaError(e.to_string()))?;

        let ictx = input(&path).map_err(|e| AforaError::MediaError(e.to_string()))?;

        let stream = ictx
            .streams()
            .best(Type::Video)
            .ok_or_else(|| AforaError::MediaError("No se encontró stream de video".into()))?;
        let video_stream_index = stream.index();

        let context_decoder =
            ffmpeg::codec::context::Context::from_parameters(stream.parameters())
                .map_err(|e| AforaError::MediaError(e.to_string()))?;

        let decoder = context_decoder
            .decoder()
            .video()
            .map_err(|e| AforaError::MediaError(e.to_string()))?;

        let scaler = ScalingContext::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            Pixel::RGB24,
            decoder.width(),
            decoder.height(),
            Flags::BILINEAR,
        )
            .map_err(|e| AforaError::MediaError(e.to_string()))?;

        Ok(Self {
            ictx,
            decoder,
            scaler,
            video_stream_index,
            frame_queue: VecDeque::new(),
            eof_sent: false,
            finished: false,
        })
    }

    /// Drena todos los frames disponibles del decoder tras un send_packet/send_eof
    /// y los empuja ya convertidos a `Frame` en la cola interna.
    fn drain_decoder(&mut self) -> Result<(), AforaError> {
        let mut decoded = FfmpegVideoFrame::empty();
        while self.decoder.receive_frame(&mut decoded).is_ok() {
            let mut rgb_frame = FfmpegVideoFrame::empty();
            self.scaler
                .run(&decoded, &mut rgb_frame)
                .map_err(|e| AforaError::MediaError(e.to_string()))?;

            self.frame_queue.push_back(ffmpeg_frame_to_frame(&rgb_frame));
        }
        Ok(())
    }

    pub fn width(&self) -> u32 {
        self.decoder.width()
    }

    pub fn height(&self) -> u32 {
        self.decoder.height()
    }

    /// Redondea el frame_rate del stream a un u32. Si ffmpeg no reporta
    /// frame_rate (0/0, típico en algunos contenedores), cae a 30 por defecto.
    pub fn fps(&self) -> u32 {
        let rate = self.decoder.frame_rate();
        match rate {
            Some(r) if r.denominator() != 0 => {
                (r.numerator() as f64 / r.denominator() as f64).round() as u32
            }
            _ => 30,
        }
    }
}

/// Copia respetando el linesize/stride real del plane RGB24 de FFmpeg,
/// que casi nunca coincide con width * 3.
fn ffmpeg_frame_to_frame(rgb_frame: &FfmpegVideoFrame) -> Frame {
    let width = rgb_frame.width();
    let height = rgb_frame.height();
    let stride = rgb_frame.stride(0);
    let plane = rgb_frame.data(0);
    let row_bytes = width as usize * 3;

    let mut data = Vec::with_capacity(row_bytes * height as usize);
    for y in 0..height as usize {
        let start = y * stride;
        data.extend_from_slice(&plane[start..start + row_bytes]);
    }

    Frame { width, height, data }
}

impl Iterator for VideoSource {
    type Item = Result<Frame, AforaError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(frame) = self.frame_queue.pop_front() {
                return Some(Ok(frame));
            }

            if self.finished {
                return None;
            }

            // OJO: `self.ictx.packets()` crea un iterador temporal cada vez.
            // Esto es seguro porque el estado de lectura vive en el AVFormatContext
            // subyacente, no en el wrapper Rust — es el patrón estándar para
            // evitar el problema de "iterador auto-referencial" contra &mut self.
            match self.ictx.packets().next() {
                Some((stream, packet)) => {
                    if stream.index() == self.video_stream_index {
                        if let Err(e) = self.decoder.send_packet(&packet) {
                            return Some(Err(AforaError::MediaError(e.to_string())));
                        }
                        if let Err(e) = self.drain_decoder() {
                            return Some(Err(e));
                        }
                    }
                    // Si el paquete es de otro stream (audio, etc.) lo ignoramos
                    // y el loop simplemente pide el siguiente.
                }
                None => {
                    if !self.eof_sent {
                        self.eof_sent = true;
                        if let Err(e) = self.decoder.send_eof() {
                            return Some(Err(AforaError::MediaError(e.to_string())));
                        }
                        if let Err(e) = self.drain_decoder() {
                            return Some(Err(e));
                        }
                    } else {
                        self.finished = true;
                    }
                }
            }
        }
    }
}