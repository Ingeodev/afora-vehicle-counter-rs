use std::path::PathBuf;
use std::sync::Arc;
use crate::core::afora_error::AforaError;
use crate::features::detector::{DetectorFactory, ModelChoice, RuntimeChoice};
use crate::features::media_source::adapters::image_source::ImageSource;
use crate::features::media_source::adapters::video_source::VideoSource;
use crate::features::tracker::adapters::oc_sort_tracker::OcSortTracker;
use crate::features::tracker::domain::tracking_input::TrackingInput;
use crate::features::tracker::ports::tracker::Tracker;
use crate::features::writter::adapters::image_writter::ImageWriter;
use crate::features::writter::adapters::video_writter::VideoWriter;

pub mod features;
pub mod core;
mod shared;

fn run_image(args: &CliArgs) -> Result<(), AforaError> {
    let mut detector = DetectorFactory::build(
        RuntimeChoice::Onnx {
            model_path: args.model_path.clone(),
            num_threads: 4,
        },
        ModelChoice::YoloOnnx {
            conf_threshold: 0.25,
        },
    )?;
    let mut tracker = OcSortTracker::new(30, 1, 0.3, 3, 0.2);

    let source = ImageSource::new(args.image_path.clone());
    let writer = ImageWriter::new()?;

    for frame in source {
        let frame = Arc::new(frame?);
        let detections = detector.detect(&frame)?;
        let tracks = tracker.update(TrackingInput {
            frame: frame.clone(),
            detections,
        })?;

        println!("Detected {} objects", tracks.len());
        writer.write(&frame, &tracks, "assets/images/output.png")?;

        for track in &tracks {
            println!("{:#?}", track);
        }
    }

    Ok(())
}

fn run_video(args: &CliArgs) -> Result<(), AforaError> {
    let mut detector = DetectorFactory::build(
        RuntimeChoice::Onnx {
            model_path: args.model_path.clone(),
            num_threads: 4,
        },
        ModelChoice::YoloOnnx {
            conf_threshold: 0.25,
        },
    )?;
    let mut tracker = OcSortTracker::new(30, 1, 0.3, 3, 0.2);

    let source = VideoSource::new(args.image_path.clone())?;

    // El writer necesita conocer width/height/fps ANTES del primer frame,
    // por eso los leemos del source recién construido (aún no consumido).
    let mut writer = VideoWriter::new(
        "assets/videos/output.mp4",
        source.width(),
        source.height(),
        source.fps(),
        23, // crf
    )?;

    for frame in source {
        let frame = Arc::new(frame?);
        let detections = detector.detect(&frame)?;
        let tracks = tracker.update(TrackingInput {
            frame: frame.clone(),
            detections,
        })?;

        println!("Detected {} objects", tracks.len());
        writer.write(&frame, &tracks)?;

        for track in &tracks {
            println!("{:#?}", track);
        }
    }

    // Imprescindible: hace flush del encoder y escribe el trailer del mp4.
    writer.finish()?;

    Ok(())
}

fn main() -> Result<(), AforaError> {
    let args = CliArgs::parse()?;

    // Asumo dispatch por extensión del archivo de entrada ya que tu CliArgs
    // actual no trae un flag explícito de modo. Si prefieres un flag
    // explícito (--mode image|video) en vez de inferirlo, lo ajustamos.
    let is_video = args
        .image_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| matches!(ext.to_lowercase().as_str(), "mp4" | "mkv" | "avi" | "mov"))
        .unwrap_or(false);

    if is_video {
        run_video(&args)
    } else {
        run_image(&args)
    }
}


struct CliArgs {
    pub image_path: PathBuf,
    pub model_path: String,
}

impl CliArgs {
    fn parse() -> Result<Self, AforaError> {

        let mut image = None;
        let mut model = None;

        let mut args = std::env::args().skip(1);

        while let Some(arg) = args.next() {

            match arg.as_str() {

                "--image" => {
                    image = args.next();
                }

                "--model" => {
                    model = args.next();
                }

                _ => {}
            }
        }

        Ok(Self {
            image_path: image.ok_or_else(|| {
                AforaError::InvalidArgument(
                    "missing --image".into()
                )
            })?.parse().unwrap(),

            model_path: model.ok_or_else(|| {
                AforaError::InvalidArgument(
                    "missing --model".into()
                )
            })?,
        })
    }
}
