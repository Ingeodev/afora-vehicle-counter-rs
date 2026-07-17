use std::path::PathBuf;
use crate::core::afora_error::AforaError;
use crate::features::detector::domain::preprocess_fallback::PreprocessFallbackPolicy;
use crate::features::detector::ports::inference_runtime::InferenceRuntimeConfig;
use crate::features::detector::ports::postprocessor::PostprocessorConfig;
use crate::features::detector::ports::preprocessor::PreprocessorConfig;
use crate::features::media_source::media_source_factory::MediaSourceChoice;
use crate::features::pipeline::domain::pipeline_config::ExecutionMode;
use crate::features::pipeline::pipeline_builder::PipelineBuilder;
use crate::features::tracker::tracker_factory::TrackerChoice;
use crate::features::tracking_suscribers::tracking_subscriber_factory::TrackerSubscriberChoice;
use crate::shared::utilities::get_video_props::get_video_properties;

pub mod features;
pub mod core;
mod shared;
mod feature_validation;

fn main() -> Result<(), AforaError> {

    //let args = CliArgs::parse()?;
    
    let args: CliArgs = CliArgs{
        source: PathBuf::from("assets/videos/video.mp4"),
            model_path:  PathBuf::from("assets/models/yolo11s_dyn.onnx"),
            max_frames: Some(100),
            video_output_path: PathBuf::from("assets/videos/output1111.mp4"),
            batch_size: 1,
            debug: true,
        debug_tags: None,
    };

    crate::shared::debug::set_debug(args.debug);
    if args.debug {
        crate::shared::stacktrace::init(args.debug_tags.as_deref());
    }

    let mut pipeline_builder = PipelineBuilder::new();

    let video_props = get_video_properties(args.source.clone())
        .map_err(|e| {
            AforaError::MediaError(String::from("El video parece encontrarse corrupto"))
        })?;

    let mut pipeline =pipeline_builder
        .set_execution_mode(ExecutionMode::Sequential)
        .set_media_source(MediaSourceChoice::Video {
            path: args.source.clone(),
            max_frames: args.max_frames,
        })?
        .set_runtime(InferenceRuntimeConfig {
            model_path: args.model_path.clone(),
            num_threads: 4,
        })
        .set_preprocessor_config(PreprocessorConfig::new(
            PreprocessFallbackPolicy::Cpu,
        ))
        .set_postprocessor_config(PostprocessorConfig {
            input_side: 640,
            batch_size: args.batch_size,
            #[cfg(feature = "yolo11")]
            conf_threshold: 0.5,
            #[cfg(feature = "yolo11")]
            nms_iou_threshold: 0.45,
        })
        .set_tracker_config(TrackerChoice::OcSort {
           max_age: 30,
           min_hits: 1,
           iou_threshold: 0.3,
           delta_t: 3,
           inertia: 0.2
        })?
        .add_subscriber(TrackerSubscriberChoice::Logger)
        .add_subscriber(TrackerSubscriberChoice::VideoWriter {
            output_path: args.video_output_path,
            width: video_props.width,
            height: video_props.height,
            fps: video_props.fps,
            crf: 23,
        })
        .build()?;

    let result = pipeline.run();
    if args.debug {
        let _ = crate::shared::stacktrace::flush_csv("stacktrace.csv");
    }
    if let Err(err) = result {
        println!("Error: {}", err);
    }

    Ok(())
}


struct CliArgs {
    pub source: PathBuf,
    pub model_path: PathBuf,
    pub max_frames: Option<i32>,
    pub video_output_path: PathBuf,
    pub debug: bool,
    pub debug_tags: Option<String>,
    batch_size: u32,
}

impl CliArgs {
    fn parse() -> Result<Self, AforaError> {

        let mut source = None;
        let mut model = None;
        let mut video_output_path = None;
        let mut max_frames: Option<i32> = None;
        let mut batch_size: u32 = 1;
        let mut debug = false;
        let mut debug_tags: Option<String> = None;

        let mut args = std::env::args().skip(1).peekable();

        while let Some(arg) = args.next() {

            match arg.as_str() {

                "--source" => {
                    source = args.next();
                }

                "--model" => {
                    model = args.next();
                }

                "--max_frames" => {
                    max_frames = args
                        .next()
                        .and_then(|s| s.parse().ok());
                }

                "--video_output_path" => {
                    video_output_path = args.next();
                }

                "--batch_size" => {
                    if let Some(arg) = args.next() {
                        if let Ok(num) = arg.parse() {
                            batch_size = num;
                        }
                    }
                }

                "--debug" => {
                    debug = true;
                    if let Some(next) = args.peek() {
                        if !next.starts_with("--") {
                            debug_tags = args.next();
                        }
                    }
                }

                _ => {}
            }
        }

        Ok(Self {
            source: source.ok_or_else(|| {
                AforaError::InvalidArgument(
                    "missing --source".into()
                )
            })?.parse().unwrap(),

            model_path: model.ok_or_else(|| {
                AforaError::InvalidArgument(
                    "missing --model".into()
                )
            })?.parse().unwrap(),

            video_output_path: video_output_path.ok_or_else(|| {
                AforaError::InvalidArgument(
                    "missing --model".into()
                )
            })?.parse().unwrap(),

            debug,
            debug_tags,

            batch_size,

            max_frames,
        })
    }
}
