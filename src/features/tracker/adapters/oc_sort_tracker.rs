use trackforge::trackers::ocsort::{OcSort, OcSortTrack};
use crate::core::afora_error::AforaError;
use crate::features::detector::domain::detection::{BoundingBox, Detection};
use crate::features::detector::ports::tensor_input::TensorInput;
use crate::features::tracker::domain::tracking_input::TrackingInput;
use crate::features::tracker::domain::tracking_output::TrackingOutput;
use crate::features::tracker::ports::tracker::Tracker;

pub struct OcSortTracker {
    instance: OcSort
}

impl OcSortTracker {
    pub fn new(    max_age: usize,
                   min_hits: usize,
                   iou_threshold: f32,
                   delta_t: usize,
                   inertia: f32,) -> Self {
        Self {
            instance: OcSort::new(max_age, min_hits, iou_threshold, delta_t, inertia)
        }
    }

    fn map_detection_domain_to_trackforge(detections: &[Detection]) -> Vec<([f32; 4], f32, i64)>{
        detections
            .iter()
            .map(|d| {
                (
                    [
                        d.bbox.x1,
                        d.bbox.y1,
                        d.bbox.x2 - d.bbox.x1,
                        d.bbox.y2 - d.bbox.y1,
                    ],
                    d.confidence,
                    d.class_id as i64,
                )
            })
            .collect()
    }

    fn oc_track_to_tracking_output(ocr_tracks: Vec<OcSortTrack>) -> Vec<TrackingOutput> {
        ocr_tracks
            .iter()
            .map(Self::map_track)
            .collect()
    }

    fn map_track(track: &OcSortTrack) -> TrackingOutput {
        let [x, y, w, h] = track.tlwh;

        TrackingOutput {
            id: track.track_id,
            bbox: BoundingBox {
                x1: x,
                y1: y,
                x2: x + w,
                y2: y + h,
            },
            class_id: track.class_id as u32,
            confidence: track.score,
        }
    }
}

impl Tracker for OcSortTracker {
    fn update(&mut self, tracking_input: TrackingInput) -> Result<Vec<TrackingOutput>, AforaError> {
        let detections = Self::map_detection_domain_to_trackforge(&tracking_input.detections);
        let tracks = self.instance.update(detections);
        let tracks = Self::oc_track_to_tracking_output(tracks);

        Ok(tracks)
    }

}