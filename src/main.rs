use std::sync::Arc;
use crate::core::afora_error::AforaError;
use crate::features::detector::{DetectorFactory, ModelChoice, RuntimeChoice};
use crate::features::media_source::adapters::image_source::ImageSource;
use crate::features::tracker::adapters::oc_sort_tracker::OcSortTracker;
use crate::features::tracker::domain::tracking_input::TrackingInput;
use crate::features::tracker::ports::tracker::Tracker;
use crate::features::writter::ImageWriter;

pub mod features;
pub mod core;
mod shared;

fn main() -> Result<(), AforaError> {
    let args = CliArgs::parse()?;

    let mut detector = DetectorFactory::build(

        RuntimeChoice::Onnx {
            model_path: args.model_path,
            num_threads: 4,
        },

        ModelChoice::YoloOnnx {
            conf_threshold: 0.25,
        },

    )?;

    let mut tracker = OcSortTracker::new(
        30, 1, 0.3, 3, 0.2
    );

    let source = ImageSource::new(args.image_path);

    for frame in source {
        let frame = Arc::new(frame?);


        let detections = detector.detect(&frame)?;

        let tracks = tracker.update(
            TrackingInput {
                frame: frame.clone(),
                detections
            }
        )?;

        println!("Detected {} objects", tracks.len());

        let writer = ImageWriter;
        writer.write(
            &frame,
            &tracks,
            "assets/images/output.png",
        )?;

        for track in &tracks {
            println!("{:#?}", track);
        }
        
    }

    Ok(())
}


struct CliArgs {
    pub image_path: String,
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
            })?,

            model_path: model.ok_or_else(|| {
                AforaError::InvalidArgument(
                    "missing --model".into()
                )
            })?,
        })
    }
}
