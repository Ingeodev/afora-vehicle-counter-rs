use crate::core::afora_error::AforaError;
use crate::features::tracker::adapters::oc_sort_tracker::OcSortTracker;
use crate::features::tracker::ports::tracker::Tracker;


pub enum TrackerChoice {
    OcSort {
        max_age: usize,
        min_hits: usize,
        iou_threshold: f32,
        delta_t: usize,
        inertia: f32,
    },
}

pub struct TrackerFactory;

impl TrackerFactory {
    pub fn build(
        tracker: TrackerChoice,
    ) -> Result<Box<dyn Tracker>, AforaError> {
        match tracker {
            TrackerChoice::OcSort {
                max_age,
                min_hits,
                iou_threshold,
                delta_t,
                inertia,
            } => Ok(Box::new(
                OcSortTracker::new(
                    max_age,
                    min_hits,
                    iou_threshold,
                    delta_t,
                    inertia,
                ),
            )),
        }
    }
}